from chimerax.core.toolshed import BundleAPI


class _MyAPI(BundleAPI):
    api_version = 1

    @staticmethod
    def initialize(session, bi):
        from . import cmd
        cmd.register_command(session)

    @staticmethod
    def start_tool(session, bi, ti):
        from .tool import FSCTool
        return FSCTool(session, ti.name)


bundle_api = _MyAPI()
