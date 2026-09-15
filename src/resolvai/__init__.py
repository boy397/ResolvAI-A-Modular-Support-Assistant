import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

# Real tweet text is full of emoji. Windows' default console/file codepage (cp1252)
# can't encode it and raises UnicodeEncodeError mid-run -- force UTF-8 everywhere
# (and replace anything the console still can't render) so logging a customer
# message never crashes the pipeline.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("resolvaiLogger")
