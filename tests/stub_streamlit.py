"""
A tiny stand-in for the `streamlit` module so app.py can be executed headlessly in tests.
It records what the page renders. It only catches wiring mistakes - it is NOT the real Streamlit.
"""
import contextlib
import types


class _State(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


class FakeUpload:
    def __init__(self, name, data):
        self.name, self._data, self.size = name, data, len(data)

    def getbuffer(self):
        return memoryview(self._data)

    def getvalue(self):
        return self._data


class _Ctx:
    def __init__(self, st, label=None):
        self._st, self.label = st, label

    def __enter__(self):
        return self._st

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):          # columns / sidebar expose the same API (col.download_button ...)
        return getattr(self._st, name)


class _Bar:
    def __init__(self, st):
        self.st = st

    def progress(self, value, text=""):
        assert 0.0 <= value <= 1.0, f"progress out of range: {value}"
        self.st.progress_values.append(value)

    def empty(self):
        self.st.progress_values.append("cleared")


class _Cfg:
    """st.column_config.XColumn(...) -> just remember the arguments."""
    def __getattr__(self, name):
        return lambda *a, **k: (name, a, k)


class RerunRequested(Exception):
    pass


def make_stub():
    st = types.ModuleType("streamlit")
    st.session_state = _State()
    st.log, st.downloads, st.progress_values = [], [], []
    st.uploads, st.press, st.selected = [], set(), None
    st.buttons = {}                                    # label -> disabled?
    st.html_blocks = []
    st.tables, st.charts = [], []                      # st.dataframe rows / st.line_chart data
    st.reruns = 0
    st.fragments = []                                  # run_every values of fragments
    st.radio_value, st.folder, st.toggle_values, st.slider_values = None, "", {}, {}

    def rec(kind):
        def f(*a, **k):
            st.log.append((kind, str(a[0]) if a else ""))
        return f

    for kind in ["caption", "title", "subheader", "write", "info", "warning", "error", "success", "set_page_config", "divider"]:
        setattr(st, kind, rec(kind))

    def markdown(body, unsafe_allow_html=False, **k):
        st.log.append(("markdown", body))
        if unsafe_allow_html and not body.lstrip().startswith("<style"):
            st.html_blocks.append(body)
    st.markdown = markdown

    st.cache_resource = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    st.cache_data = st.cache_resource
    st.expander = lambda *a, **k: _Ctx(st)
    st.container = lambda *a, **k: _Ctx(st)
    st.spinner = lambda *a, **k: _Ctx(st)
    st.sidebar = _Ctx(st)
    st.columns = lambda spec, **k: [_Ctx(st) for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.tabs = lambda labels, **k: [_Ctx(st) for _ in labels]
    st.column_config = _Cfg()

    def fragment(func=None, run_every=None, **k):
        st.fragments.append(run_every)
        return (lambda f: f) if func is None else func
    st.fragment = fragment

    def rerun(*a, **k):
        st.reruns += 1                                 # the real one stops the script; here the test re-runs it by hand
    st.rerun = rerun

    def progress(value, text=""):
        st.progress_values.append(value)
        return _Bar(st)
    st.progress = progress

    def button(label, *a, disabled=False, **k):
        st.buttons[label] = disabled
        return (not disabled) and any(p in label for p in st.press)
    st.button = button

    def download_button(label, data=None, file_name=None, mime=None, **k):
        st.downloads.append((label, file_name, len(data) if data is not None else 0))
        return False
    st.download_button = download_button

    def selectbox(label, options, **k):
        options = list(options)
        st.log.append(("selectbox", ",".join(options)))
        return st.selected if st.selected in options else options[0]
    st.selectbox = selectbox
    st.file_uploader = lambda label, **k: st.uploads if k.get("accept_multiple_files") else None

    def radio(label, options, **k):
        options = list(options)
        return st.radio_value if st.radio_value in options else options[0]
    st.radio = radio
    st.text_input = lambda label, **k: st.folder
    st.toggle = lambda label, value=False, **k: st.toggle_values.get(label, value)
    st.slider = lambda label, min_value=None, max_value=None, value=None, step=None, **k: st.slider_values.get(label, value)

    def dataframe(data, **k):
        st.tables.append(data)
    st.dataframe = dataframe

    def line_chart(data, **k):
        st.charts.append(data)
    st.line_chart = line_chart
    return st
