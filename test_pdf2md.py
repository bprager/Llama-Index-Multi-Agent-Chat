#!/usr/bin/env python3
"""
Convert all pdf documents in the current directory to markdown files.
"""
import logging
import time
import pathlib

from llama_index.core import SimpleDirectoryReader, schema
from llama_parse import LlamaParse, ResultType  # type: ignore [import-untyped]
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# Settings
class ModelSettings(BaseSettings):
    """Settings for knowledge graph builder"""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8",
    )
    llama_cloud_api_key: SecretStr
    logging_level: str = "INFO"


settings = ModelSettings()  # type: ignore [call-arg]

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.logging_level),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(funcName)s:%(lineno)d: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("to_markdowns.log", mode="w"),
    ],
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Loggers exlude from debug
logging.getLogger("asyncio").setLevel(logging.INFO)
logging.getLogger("fsspec").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.INFO)

# set up parser
parser = LlamaParse(
    result_type=ResultType.MD,
    split_by_page=False,  # force to split by page
    api_key=settings.llama_cloud_api_key.get_secret_value(),
)


def main():
    """Main function"""

    logging.debug("Converting pdf files to markdown")
    file_extractor = {".pdf": parser}
    documents: list[schema.Document] = []  # type: ignore [annotation-unchecked]
    documents = SimpleDirectoryReader(".", file_extractor=file_extractor).load_data()
    for doc in documents:
        file_name: str = doc.metadata["file_name"]  # type: ignore [annotation-unchecked]
        if file_name.endswith(".pdf"):
            logging.debug("Converting %s", file_name)
            base_name: str = pathlib.Path(file_name).stem  # type: ignore [annotation-unchecked]
            markdown_file_name: str  # type: ignore [annotation-unchecked]
            markdown_file_name = f"{base_name}.md"
            with open(markdown_file_name, mode="w", encoding="utf-8") as f:
                f.write(doc.text)

    logging.debug("Done")


if __name__ == "__main__":
    start_time = time.time()
    main()
    mins, secs = divmod(time.time() - start_time, 60)
    hrs, mins = divmod(mins, 60)
    logging.info("Execution time: %02d:%02d:%02d", hrs, mins, secs)
