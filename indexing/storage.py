
import hashlib

class PdfBytesStore:
    def store(self,pdf_bytes: bytes,file_hash:str)-> str:
        """
                将 PDF 字节保存为临时文件
                :param pdf_bytes: PDF 的原始字节数据
                :param file_hash: 文件内容的 MD5 哈希值（用于命名）
                :return: 临时文件的路径
        """
        temp_path=f"temp_{file_hash}.pdf"
        with open(temp_path, "wb") as f:
            f.write(pdf_bytes)
        return temp_path

    def compute_hash(self,pdf_bytes)-> str:
        return hashlib.md5(pdf_bytes).hexdigest()