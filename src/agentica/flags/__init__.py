from os import getenv

in_testing = getenv("AGENTICA_INTEGRATION_TESTING") == "1"
