class AnalyzerException(Exception):
    pass


class InvalidR2KeyError(AnalyzerException):
    pass


class DownloadError(AnalyzerException):
    pass


class AnalysisError(AnalyzerException):
    pass


class CallbackError(AnalyzerException):
    pass
