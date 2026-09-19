import os
import pickle
import threading

import pytest

from jinja2 import Environment
from jinja2.bccache import bc_magic
from jinja2.bccache import Bucket
from jinja2.bccache import FileSystemBytecodeCache
from jinja2.bccache import MemcachedBytecodeCache
from jinja2.exceptions import TemplateNotFound


@pytest.fixture
def env(package_loader, tmp_path):
    bytecode_cache = FileSystemBytecodeCache(str(tmp_path))
    return Environment(loader=package_loader, bytecode_cache=bytecode_cache)


def make_code():
    return compile("result = 'cached'", "<template>", "exec")


class TestByteCodeCache:
    def test_simple(self, env):
        tmpl = env.get_template("test.html")
        assert tmpl.render().strip() == "BAR"
        pytest.raises(TemplateNotFound, env.get_template, "missing.html")

    def test_atomic_dump_keeps_filename_and_checksum(self, tmp_path, monkeypatch):
        cache = FileSystemBytecodeCache(str(tmp_path))
        bucket = Bucket(None, "key", "checksum")
        bucket.code = make_code()
        filename = cache._get_cache_filename(bucket)
        assert filename == os.path.join(str(tmp_path), "__jinja2_key.cache")
        replace = os.replace
        temporary_names = []

        def assert_same_directory_replace(source, destination):
            temporary_names.append(source)
            assert os.path.dirname(source) == os.path.dirname(destination)
            assert os.path.exists(source)
            replace(source, destination)

        monkeypatch.setattr(os, "replace", assert_same_directory_replace)

        cache.dump_bytecode(bucket)

        assert temporary_names
        assert os.path.exists(filename)
        assert not any(os.path.exists(name) for name in temporary_names)

        loaded = Bucket(None, "key", "checksum")
        cache.load_bytecode(loaded)
        assert loaded.code is not None

        stale = Bucket(None, "key", "different checksum")
        cache.load_bytecode(stale)
        assert stale.code is None

    @pytest.mark.parametrize("failure", ["write", "replace", "cancel"])
    def test_dump_failure_cleans_temporary_file(
        self, tmp_path, monkeypatch, failure
    ):
        cache = FileSystemBytecodeCache(str(tmp_path))
        bucket = Bucket(None, "key", "checksum")
        bucket.code = make_code()
        removed_temporary_names = []
        real_remove = os.remove

        def track_remove(path, *args, **kwargs):
            removed_temporary_names.append(path)
            return real_remove(path, *args, **kwargs)

        if failure == "write":
            def fail_write(f):
                f.write(b"partial")
                raise RuntimeError("write failed")

            expected = RuntimeError
            monkeypatch.setattr(bucket, "write_bytecode", fail_write)
        elif failure == "replace":
            def fail_replace(source, destination):
                raise OSError("replace failed")

            expected = OSError
            monkeypatch.setattr(os, "replace", fail_replace)
        else:
            def cancel_write(f):
                f.write(b"partial")
                raise KeyboardInterrupt

            expected = KeyboardInterrupt
            monkeypatch.setattr(bucket, "write_bytecode", cancel_write)

        monkeypatch.setattr(os, "remove", track_remove)

        with pytest.raises(expected):
            cache.dump_bytecode(bucket)

        assert removed_temporary_names
        assert all(not os.path.exists(name) for name in removed_temporary_names)
        assert not list(tmp_path.glob("*.tmp"))

    def test_concurrent_read_and_write(self, tmp_path):
        cache = FileSystemBytecodeCache(str(tmp_path))
        errors = []

        def writer():
            try:
                for _ in range(30):
                    bucket = Bucket(None, "key", "checksum")
                    bucket.code = make_code()
                    cache.dump_bytecode(bucket)
            except BaseException as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    bucket = Bucket(None, "key", "checksum")
                    cache.load_bytecode(bucket)
                    if bucket.code is not None:
                        exec(bucket.code, {})
            except BaseException as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(7)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        loaded = Bucket(None, "key", "checksum")
        cache.load_bytecode(loaded)
        assert loaded.code is not None

    def test_clear_during_read_and_write(self, tmp_path):
        cache = FileSystemBytecodeCache(str(tmp_path))
        errors = []

        def writer():
            try:
                for _ in range(50):
                    bucket = Bucket(None, "key", "checksum")
                    bucket.code = make_code()
                    cache.dump_bytecode(bucket)
            except BaseException as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    bucket = Bucket(None, "key", "checksum")
                    cache.load_bytecode(bucket)
            except BaseException as e:
                errors.append(e)

        def clearer():
            try:
                for _ in range(50):
                    cache.clear()
            except BaseException as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=clearer),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []

    def test_clear_handles_removal_races(self, tmp_path, monkeypatch):
        cache = FileSystemBytecodeCache(str(tmp_path))
        bucket = Bucket(None, "key", "checksum")
        bucket.code = make_code()
        cache.dump_bytecode(bucket)
        filename = cache._get_cache_filename(bucket)

        def missing_error():
            return FileNotFoundError(2, "No such file or directory")

        def sharing_violation():
            monkeypatch.setattr(os, "name", "nt")
            error = PermissionError(13, "Permission denied")
            error.winerror = 32
            return error

        def directory_error():
            return IsADirectoryError(21, "Is a directory")

        def permission_error():
            return PermissionError(13, "Permission denied")

        cases = [
            (missing_error, False),
            (sharing_violation, False),
            (directory_error, False),
            (permission_error, True),
        ]

        for make_error, should_raise in cases:
            monkeypatch.undo()

            def fail_remove(path, *args, make_error=make_error, **kwargs):
                assert path == filename
                raise make_error()

            monkeypatch.setattr(os, "remove", fail_remove)

            if should_raise:
                with pytest.raises(PermissionError):
                    cache.clear()
            else:
                cache.clear()

    def test_missing_file_and_directory_placeholder_are_cache_misses(self, tmp_path):
        cache = FileSystemBytecodeCache(str(tmp_path))

        missing = Bucket(None, "missing", "checksum")
        cache.load_bytecode(missing)
        assert missing.code is None

        directory_bucket = Bucket(None, "directory", "checksum")
        os.mkdir(cache._get_cache_filename(directory_bucket))
        cache.load_bytecode(directory_bucket)
        assert directory_bucket.code is None

        cache.clear()
        assert os.path.isdir(cache._get_cache_filename(directory_bucket))

    @pytest.mark.parametrize(
        "winerror, stat_raises",
        [(32, False), (5, True)],
        ids=["sharing_violation", "delete_pending"],
    )
    def test_windows_removal_errors_are_cache_misses(
        self, tmp_path, monkeypatch, winerror, stat_raises
    ):
        cache = FileSystemBytecodeCache(str(tmp_path))
        filename = cache._get_cache_filename(Bucket(None, "key", "checksum"))
        real_open = open
        real_stat = os.stat
        monkeypatch.setattr(os, "name", "nt")

        def fail_open(path, *args, **kwargs):
            if path == filename:
                error = PermissionError(
                    winerror, "Permission denied", None, winerror
                )
                raise error
            return real_open(path, *args, **kwargs)

        def maybe_fail_stat(path, *args, **kwargs):
            if stat_raises and path == filename:
                error = PermissionError(
                    winerror, "Permission denied", None, winerror
                )
                raise error
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fail_open)
        monkeypatch.setattr(os, "stat", maybe_fail_stat)

        bucket = Bucket(None, "key", "checksum")
        cache.load_bytecode(bucket)

        assert bucket.code is None

    def test_windows_directory_permission_error_is_cache_miss(
        self, tmp_path, monkeypatch
    ):
        cache = FileSystemBytecodeCache(str(tmp_path))
        bucket = Bucket(None, "key", "checksum")
        filename = cache._get_cache_filename(bucket)
        os.mkdir(filename)
        real_open = open
        monkeypatch.setattr(os, "name", "nt")

        def fail_open(path, *args, **kwargs):
            if path == filename:
                error = PermissionError(5, "Access denied")
                error.winerror = 5
                raise error
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fail_open)

        loaded = Bucket(None, "key", "checksum")
        cache.load_bytecode(loaded)

        assert loaded.code is None

    def test_real_permission_errors_are_not_hidden(self, tmp_path, monkeypatch):
        cache = FileSystemBytecodeCache(str(tmp_path))
        bucket = Bucket(None, "key", "checksum")
        filename = cache._get_cache_filename(bucket)
        real_open = open

        def fail_open(path, *args, **kwargs):
            if path == filename:
                raise PermissionError(13, "Permission denied")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fail_open)

        with pytest.raises(PermissionError):
            cache.load_bytecode(Bucket(None, "key", "checksum"))

        error = PermissionError(13, "Permission denied", None, 32)
        error.winerror = 32

        def fail_with_windows_error(path, *args, **kwargs):
            if path == filename:
                raise error
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fail_with_windows_error)

        with pytest.raises(PermissionError):
            cache.load_bytecode(Bucket(None, "key", "checksum"))

        monkeypatch.undo()
        real_open = open

        def fail_existing_open(path, *args, **kwargs):
            if path == filename:
                raise PermissionError(13, "Permission denied")
            return real_open(path, *args, **kwargs)

        with open(filename, "wb"):
            pass
        monkeypatch.setattr(os, "name", "nt")
        monkeypatch.setattr("builtins.open", fail_existing_open)

        with pytest.raises(PermissionError):
            cache.load_bytecode(Bucket(None, "key", "checksum"))

    @pytest.mark.skipif(
        not hasattr(os, "geteuid") or os.geteuid() == 0,
        reason="permission test requires an unprivileged POSIX user",
    )
    def test_real_filesystem_permission_error_is_raised(self, tmp_path):
        cache = FileSystemBytecodeCache(str(tmp_path))
        bucket = Bucket(None, "key", "checksum")
        bucket.code = make_code()
        cache.dump_bytecode(bucket)
        filename = cache._get_cache_filename(bucket)
        os.chmod(filename, 0)

        try:
            with pytest.raises(PermissionError):
                cache.load_bytecode(Bucket(None, "key", "checksum"))
        finally:
            os.chmod(filename, 0o644)

    def test_truncated_marshal_data_is_cache_miss(self, tmp_path):
        cache = FileSystemBytecodeCache(str(tmp_path))
        bucket = Bucket(None, "key", "checksum")
        filename = cache._get_cache_filename(bucket)
        with open(filename, "wb") as f:
            f.write(bc_magic)
            pickle.dump(bucket.checksum, f, 2)
            f.write(b"not marshal data")

        cache.load_bytecode(bucket)

        assert bucket.code is None

    def test_deserialization_error_is_not_hidden(self, tmp_path):
        cache = FileSystemBytecodeCache(str(tmp_path))
        filename = cache._get_cache_filename(Bucket(None, "key", "checksum"))
        with open(filename, "wb") as f:
            f.write(bc_magic)
            f.write(b"\x80\x02X")

        with pytest.raises(pickle.UnpicklingError):
            cache.load_bytecode(Bucket(None, "key", "checksum"))


class MockMemcached:
    class Error(Exception):
        pass

    key = None
    value = None
    timeout = None

    def get(self, key):
        return self.value

    def set(self, key, value, timeout=None):
        self.key = key
        self.value = value
        self.timeout = timeout

    def get_side_effect(self, key):
        raise self.Error()

    def set_side_effect(self, *args):
        raise self.Error()


class TestMemcachedBytecodeCache:
    def test_dump_load(self):
        memcached = MockMemcached()
        m = MemcachedBytecodeCache(memcached)

        b = Bucket(None, "key", "")
        b.code = "code"
        m.dump_bytecode(b)
        assert memcached.key == "jinja2/bytecode/key"

        b = Bucket(None, "key", "")
        m.load_bytecode(b)
        assert b.code == "code"

    def test_exception(self):
        memcached = MockMemcached()
        memcached.get = memcached.get_side_effect
        memcached.set = memcached.set_side_effect
        m = MemcachedBytecodeCache(memcached)
        b = Bucket(None, "key", "")
        b.code = "code"

        m.dump_bytecode(b)
        m.load_bytecode(b)

        m.ignore_memcache_errors = False

        with pytest.raises(MockMemcached.Error):
            m.dump_bytecode(b)

        with pytest.raises(MockMemcached.Error):
            m.load_bytecode(b)
