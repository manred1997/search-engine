import os

from dynaconf import Dynaconf

AUTO_COMPLETE_PATH = os.environ.get("AUTO_COMPLETE_PATH")


settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=[
        os.path.join(AUTO_COMPLETE_PATH, "src/settings/settings.json"),
        os.path.join(AUTO_COMPLETE_PATH, "src/settings/.secrets.json"),
    ],
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
