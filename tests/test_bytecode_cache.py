import errno
import os
import pickle
import random
import threading

import pytest

from jinja2 import Environment
from jinja2.bccache import bc_magic
from jinja2.bccache import Bucket
from jinja2.bccache import FileSystemBytecodeCache
from jinja2.bccache import MemcachedBytecodeCache
from jinja2.exceptions import TemplateNotFound
from jinja2.loaders import DictLoader


@pytest.fixture
def env(package_loader, tmp_path):
    bytecode_cache = FileSystemBytecodeCache(str(tmp_path))
    return Environment(loader=package_loader, bytecode_cache=bytecode_cache)


class TestByteCodeCache:
    def test_simple(self, env):
        tmpl = env.get_template("test.html")
        assert tmpl.render().strip() == "BAR"
        pytest.raises(TemplateNotFound, env.get_template, "missing.html")


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


def make_bucket(tmp_path, key="key", checksum="checksum", code=1):
    cache = FileSystemBytecodeCache(str(tmp_path))
    bucket = Bucket(Environment(), key, checksum)
    if code is not None:
        bucket.code = compile(repr(code), "<cache>", "eval")
    return cache, bucket


def load_bucket(tmp_path, key="key", checksum="checksum"):
    cache = FileSystemBytecodeCache(str(tmp_path))
    bucket = Bucket(Environment(), key, checksum)
    cache.load_bytecode(bucket)
    return bucket


def eval_code(code):
    namespace = {}
    exec(code, namespace)
    return namespace


class TestFileSystemBytecodeCache:
    def test_dump_and_load_roundtrip(self, tmp_path):
        cache, bucket = make_bucket(tmp_path)
        cache.dump_bytecode(bucket)

        loaded = load_bucket(tmp_path)
        assert loaded.code is not None
        assert eval(loaded.code, {}) == 1

    def test_filename_rule_unchanged(self, tmp_path):
        cache, bucket = make_bucket(tmp_path, key="deadbeef")
        cache.dump_bytecode(bucket)
        assert (tmp_path / "__jinja2_deadbeef.cache").is_file()

    def test_load_missing_is_miss(self, tmp_path):
        loaded = load_bucket(tmp_path)
        assert loaded.code is None

    def test_load_directory_instead_of_file_is_miss(self, tmp_path):
        # A directory occupying the cache file's name (for example left
        # behind by another tool) must be treated as a miss.
        os.mkdir(tmp_path / "__jinja2_key.cache")
        loaded = load_bucket(tmp_path)
        assert loaded.code is None

    def test_load_notdir_component_is_miss(self, tmp_path):
        # A component of the cache path being a file gives ENOTDIR.
        (tmp_path / "__jinja2_key.cache").write_bytes(b"blocker")
        loaded = load_bucket(tmp_path, key="key/deep")
        assert loaded.code is None

    @pytest.mark.skipif(
        not hasattr(os, "geteuid"), reason="permission semantics only tested on POSIX"
    )
    @pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file permissions")
    def test_load_permission_error_is_raised(self, tmp_path):
        cache, bucket = make_bucket(tmp_path)
        cache.dump_bytecode(bucket)
        filename = tmp_path / "__jinja2_key.cache"
        filename.chmod(0)
        try:
            with pytest.raises(PermissionError):
                load_bucket(tmp_path)
        finally:
            filename.chmod(0o600)

    @pytest.mark.parametrize(
        ("error_errno", "expected"),
        [
            (errno.ENOENT, True),
            (errno.ENOTDIR, True),
            (errno.EISDIR, True),
            # Real permission problems on POSIX are not caused by a
            # deletion race and must surface.
            (errno.EACCES, False),
            (errno.EPERM, False),
            (errno.EROFS, False),
        ],
    )
    def test_cache_miss_error_classification_posix(self, error_errno, expected):
        from jinja2 import bccache

        assert os.name == "posix"
        error = PermissionError(error_errno, "problem")
        assert bccache._is_cache_miss_error(error) is expected

    def test_cache_miss_error_classification_windows(self):
        # The Windows branch can only run with ``os.name == "nt"``; check
        # it in an isolated process so the rest of the test machinery does
        # not see the faked platform.
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import errno, os\n"
                "from jinja2 import bccache\n"
                "os.name = 'nt'\n"
                "assert bccache._is_cache_miss_error("
                "PermissionError(errno.EACCES, 'del'))\n"
                "assert bccache._is_cache_miss_error("
                "PermissionError(errno.EPERM, 'del'))\n"
                "assert bccache._is_cache_miss_error("
                "FileNotFoundError(errno.ENOENT, 'gone'))\n"
                "assert not bccache._is_cache_miss_error("
                "PermissionError(errno.EROFS, 'ro'))\n",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_load_windows_deletion_permission_is_miss(self, tmp_path):
        # Simulate the open-time PermissionError Windows raises for a
        # file that is being deleted, with ``os.name`` faked in a
        # subprocess to avoid disturbing the host test process.
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import errno, os, sys\n"
                "os.name = 'nt'\n"
                "from jinja2.bccache import FileSystemBytecodeCache, Bucket\n"
                "from jinja2 import Environment\n"
                f"cache = FileSystemBytecodeCache({str(tmp_path)!r})\n"
                "bucket = Bucket(Environment(), 'key', 'checksum')\n"
                "bucket.code = compile('1', '<x>', 'eval')\n"
                "cache.dump_bytecode(bucket)\n"
                "real_open = open\n"
                "def deleting_open(file, mode='r', *a, **kw):\n"
                "    if str(file).endswith('__jinja2_key.cache'):\n"
                "        raise PermissionError(errno.EACCES, 'deleting')\n"
                "    return real_open(file, mode, *a, **kw)\n"
                "import jinja2.bccache as bc\n"
                "bc.builtins.open = deleting_open\n"
                "loaded = Bucket(Environment(), 'key', 'checksum')\n"
                "cache.load_bytecode(loaded)\n"
                "assert loaded.code is None\n",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_load_permission_error_while_reading_is_raised(
        self, tmp_path, monkeypatch
    ):
        cache, bucket = make_bucket(tmp_path)
        cache.dump_bytecode(bucket)

        class BrokenReader:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def read(self, size=-1):
                raise PermissionError(errno.EACCES, "access denied")

        real_open = open

        def broken_open(file, mode="r", *args, **kwargs):
            if str(file) == str(tmp_path / "__jinja2_key.cache"):
                return BrokenReader()
            return real_open(file, mode, *args, **kwargs)

        monkeypatch.setattr("jinja2.bccache.builtins.open", broken_open)
        with pytest.raises(PermissionError):
            load_bucket(tmp_path)

    def test_load_checksum_mismatch_is_miss(self, tmp_path):
        cache, bucket = make_bucket(tmp_path, checksum="old")
        cache.dump_bytecode(bucket)
        assert load_bucket(tmp_path, checksum="new").code is None

    def test_load_truncated_marshal_is_miss(self, tmp_path):
        # A corrupt/partial marshal section is silently rejected by the
        # bucket and recompilation happens.
        (tmp_path / "__jinja2_key.cache").write_bytes(
            bc_magic + pickle.dumps("checksum", 2)
        )
        assert load_bucket(tmp_path).code is None

    def test_load_truncated_checksum_is_miss(self, tmp_path):
        # A file ending right after the magic header (the classic
        # half-written cache file) is a miss, not an EOFError crash.
        (tmp_path / "__jinja2_key.cache").write_bytes(bc_magic)
        assert load_bucket(tmp_path).code is None

    def test_load_real_deserialization_error_is_raised(self, tmp_path):
        # Content after the magic header that pickle cannot parse is a
        # genuine corruption and must not be hidden.
        (tmp_path / "__jinja2_key.cache").write_bytes(bc_magic + b"\xff\xffgarbage")
        with pytest.raises(pickle.UnpicklingError):
            load_bucket(tmp_path)

    def test_dump_is_atomic(self, tmp_path):
        cache, first = make_bucket(tmp_path, code=1)
        cache.dump_bytecode(first)
        filename = tmp_path / "__jinja2_key.cache"

        class StallWriter:
            def __init__(self):
                self.started = threading.Event()
                self.proceed = threading.Event()

            def __call__(self, f):
                self.started.set()
                assert self.proceed.wait(timeout=5)
                # Replace the partial payload with a complete second
                # version so the finished cache stays valid.
                f.seek(0)
                f.truncate()
                second = make_bucket(tmp_path, code=2)[1]
                second.write_bytecode(f)

        writer = StallWriter()
        second = make_bucket(tmp_path, code=2)[1]
        second.write_bytecode = writer

        thread = threading.Thread(target=cache.dump_bytecode, args=(second,))
        thread.start()
        try:
            assert writer.started.wait(timeout=5)
            # While the replacement is in flight a reader must either hit
            # the complete old file or see nothing at the temp file --
            # never a half-written entry under the final name.
            loaded = load_bucket(tmp_path)
            assert loaded.code is None or eval(loaded.code, {}) == 1
            assert list(tmp_path.glob("__jinja2_*.cache")) == [filename]
        finally:
            writer.proceed.set()
            thread.join(5)

        assert not thread.is_alive()
        assert eval(load_bucket(tmp_path).code, {}) == 2
        assert list(tmp_path.glob("*.tmp")) == []

    def test_dump_failure_cleans_up_temp_file(self, tmp_path):
        cache, bucket = make_bucket(tmp_path)

        def broken_write(f):
            f.write(bc_magic)
            raise OSError(errno.ENOSPC, "no space left")

        bucket.write_bytecode = broken_write

        with pytest.raises(OSError):
            cache.dump_bytecode(bucket)

        assert list(tmp_path.glob("*.tmp")) == []
        assert not (tmp_path / "__jinja2_key.cache").exists()

    def test_dump_empty_bucket_cleans_up_temp_file(self, tmp_path):
        cache, bucket = make_bucket(tmp_path, code=None)

        with pytest.raises(TypeError):
            cache.dump_bytecode(bucket)

        assert list(tmp_path.glob("*.tmp")) == []

    def test_dump_cancelled_cleans_up_temp_file(self, tmp_path):
        cache, bucket = make_bucket(tmp_path)
        processed = threading.Event()

        def cancel_write(f):
            processed.set()
            raise KeyboardInterrupt

        bucket.write_bytecode = cancel_write

        with pytest.raises(KeyboardInterrupt):
            cache.dump_bytecode(bucket)

        assert list(tmp_path.glob("*.tmp")) == []

    def test_concurrent_reads_while_writing(self, tmp_path):
        cache = FileSystemBytecodeCache(str(tmp_path))
        loader = DictLoader({"t.html": "value = {{ value }}"})
        errors = []
        stop = threading.Event()

        def writer():
            for value in range(50):
                env = Environment(
                    loader=DictLoader({"t.html": f"value = {{ value }}={value}"})
                )
                env.bytecode_cache = cache
                try:
                    env.get_template("t.html")
                except Exception as e:  # pragma: no cover
                    errors.append(e)
                    stop.set()
                    return

        def reader():
            while not stop.is_set():
                env = Environment(loader=loader)
                env.bytecode_cache = cache
                try:
                    template = env.get_template("t.html")
                    assert template.render(value=7) == "value = 7"
                except Exception as e:  # pragma: no cover
                    errors.append(e)
                    return

        readers = [threading.Thread(target=reader) for _ in range(4)]
        for reader in readers:
            reader.start()
        write_thread = threading.Thread(target=writer)
        write_thread.start()
        write_thread.join(10)
        stop.set()
        for reader in readers:
            reader.join(5)

        assert errors == []
        assert not write_thread.is_alive()
        assert list(tmp_path.glob("*.tmp")) == []

    def test_concurrent_clear_and_load(self, tmp_path):
        cache = FileSystemBytecodeCache(str(tmp_path))
        loader = DictLoader({"t.html": "hello"})
        errors = []

        def worker():
            for _ in range(100):
                env = Environment(loader=loader)
                env.bytecode_cache = cache
                try:
                    template = env.get_template("t.html")
                    assert template.render() == "hello"
                    if random.random() < 0.2:
                        cache.clear()
                except Exception as e:
                    errors.append(e)
                    return

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(15)

        assert errors == []
        assert all(not thread.is_alive() for thread in threads)
        assert list(tmp_path.glob("*.tmp")) == []

    def test_clear_does_not_remove_temp_files(self, tmp_path):
        cache, bucket = make_bucket(tmp_path)
        cache.dump_bytecode(bucket)
        (tmp_path / ".__jinja2_other.cache.xyz.tmp").write_bytes(b"partial")

        cache.clear()

        assert not (tmp_path / "__jinja2_key.cache").exists()
        assert (tmp_path / ".__jinja2_other.cache.xyz.tmp").exists()
