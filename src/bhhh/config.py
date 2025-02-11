# =====================================================================================
# Check Available Packages
# =====================================================================================

try:
    import ruspy  # noqa: F401
except ImportError:
    IS_RUSPY_INSTALLED = False
else:
    IS_RUSPY_INSTALLED = True
