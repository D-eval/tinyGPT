__all__ = ["SymbolSegmenter", "SymbolSet"]


def __getattr__(name: str):
    if name in __all__:
        from .symbol_segmenter import SymbolSegmenter, SymbolSet

        return {"SymbolSegmenter": SymbolSegmenter, "SymbolSet": SymbolSet}[name]
    raise AttributeError(name)
