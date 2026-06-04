from config.setting import QaConfig
from indexing.indexing_pipeline import IndexingPipeline

config=QaConfig
pipeline=IndexingPipeline(config)
pdf_path="D:\\tools\\pythonTools\\pythonProject\\smart-reading\\data\\sample_document.pdf"
file_hash=pipeline.build_from_file(pdf_path)
print(file_hash)