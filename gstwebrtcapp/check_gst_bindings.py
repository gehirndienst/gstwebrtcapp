# run this check before importing gstapp modules
# Author: Nikita Smirnov


def check_gst():
    check_gst_global()
    check_webrtcsink()
    check_gst_elements()
    check_run_pipeline()
    print("OK: all required GStreamer objects are installed correctly!")


def check_gst_global():
    try:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version('GstWebRTC', '1.0')
        gi.require_version('GLib', '2.0')
        gi.require_version('GObject', '2.0')

        from gi.repository import Gst

        Gst.init(None)
        Gst.Pipeline.new("dummy-pipeline")
        print("CHECKING: GStreamer python bindings are installed correctly...")
    except Exception:
        raise Exception(
            "ERROR: GStreamer python plugins are not installed correctly. Please install it before running this"
            " application with apt install python3-gst-1.0 python-gi-dev python3-gi gstreamer1.0-python3-plugin-loader"
        )


def check_webrtcsink():
    try:
        import gi

        gi.require_version('Gst', '1.0')
        from gi.repository import Gst

        Gst.init(None)
        Gst.ElementFactory.make("webrtcsink", None)
        print("CHECKING: GStreamer webrtcsink is installed correctly...")
    except Exception:
        raise Exception(
            "ERROR: GStreamer webrtcsink is not installed correctly. Please install it before running this application"
            " by building the repo https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs and adding"
            " path-to-gst-plugins-rs/target/debug to the GST_PLUGIN_PATH ."
        )


def check_gst_elements():
    # operate with gst elements
    try:
        import gi

        gi.require_version('Gst', '1.0')
        from gi.repository import Gst

        Gst.init(None)
        pipeline = Gst.Pipeline.new('test-pipeline')
        source = Gst.ElementFactory.make('fakesrc', 'source')
        source.set_property('num-buffers', 5)
        sink = Gst.ElementFactory.make('fakesink', 'sink')
        pipeline.add(source)
        pipeline.add(sink)
        Gst.Element.link(source, sink)
        print("CHECKING: GStreamer elements are running correctly...")
    except Exception:
        raise Exception("ERROR: can't operate with the most important Gst.Element's!")


def check_run_pipeline():
    # run test gst pipeline for 3 seconds
    try:
        import gi

        gi.require_version('Gst', '1.0')
        gi.require_version('GLib', '2.0')
        from gi.repository import Gst, GLib

        Gst.init(None)

        pipeline_description = "videotestsrc ! autovideosink"
        pipeline = Gst.parse_launch(pipeline_description)

        print("CHECKING: start playing a test pipeline...")
        pipeline.set_state(Gst.State.PLAYING)

        def _stop_pipeline():
            pipeline.set_state(Gst.State.NULL)
            loop.quit()

        loop = GLib.MainLoop()
        GLib.timeout_add_seconds(3, _stop_pipeline)
        loop.run()
    except Exception:
        raise Exception("ERROR: can't play a simple pipeline!")


if __name__ == "__main__":
    check_gst()
