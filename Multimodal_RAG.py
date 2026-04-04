from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
import logging
from ibm_granite_community.notebook_utils import get_env_var
from langchain_community.llms import Replicate
from transformers import AutoProcessor
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
load_dotenv()
logging.basicConfig(level=logging.INFO)


embeddings_model_path = 'ibm-granite/granite-embedding-30m-english'
embeddings_model = HuggingFaceEmbeddings(

    model_name=embeddings_model_path,

)

embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)
vision_model_path = "ibm-granite/granite-vision-3.2-2b"

vision_model = Replicate(

    model=vision_model_path,
    replicate_api_token=get_env_var("REPLICATE_API_TOKEN"),
    model_kwargs={
        "max_tokens": embeddings_tokenizer.max_len_single_sentence,
        "min_tokens": 100,
    }
)

vision_processor = AutoProcessor.from_pretrained(vision_model_path)
pdf_pipeline_options = PdfPipelineOptions(
    do_ocr=False,
    generate_picture_images=True,
)

format_options = {
    InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
}

converter = DocumentConverter(format_options=format_options)
sources = [

    “https://midwestfoodbank.org/images/AR_2020_WEB2.pdf”,

]

conversions = {source: converter.convert(
    source=source).document for source in sources}
