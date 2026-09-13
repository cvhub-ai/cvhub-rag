from pathlib import Path
from typing import cast
from uuid import uuid4

import fitz
import numpy as np
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR
from rapidocr.utils.output import RapidOCROutput

from src.models.enums import FileType, ParserType, StructureSource
from src.models.parser_document import (
    Block,
    BoundingBox,
    Page,
    ParsedDocument,
    SourceInfo,
    StructureInfo,
)
from src.parsers.base import BaseDocumentParser
from src.parsers.pdf.constants import DOCLING_BLOCK_TYPE_MAP, OCR_TEXT_LABELS
from src.parsers.pdf.layout_region import PDFLayoutRegion


class PDFParser(BaseDocumentParser):
    def __init__(self) -> None:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )

        self._ocr = RapidOCR(
            params={
                "Det.engine_type": EngineType.TORCH,
                "Det.model_type": ModelType.MEDIUM,
                "Det.ocr_version": OCRVersion.PPOCRV6,
                "Cls.engine_type": EngineType.TORCH,
                "Rec.engine_type": EngineType.TORCH,
                "Rec.model_type": ModelType.MEDIUM,
                "Rec.ocr_version": OCRVersion.PPOCRV6,
                "EngineConfig.torch.use_cuda": True,
                "EngineConfig.torch.cuda_ep_cfg.device_id": 0,
            }
        )

    def parse(
        self,
        file_path: str | Path,
    ) -> ParsedDocument:
        path = Path(file_path)

        regions = self._analyze_layout(path)

        self._apply_ocr_fallback(
            path,
            regions,
        )

        pages = self._build_pages(
            regions,
        )

        return ParsedDocument(
            document_id=path.stem,
            source=SourceInfo(
                file_name=path.name,
                file_type=FileType.PDF,
                parser=ParserType.PDF,
            ),
            pages=pages,
        )

    def _analyze_layout(
        self,
        file_path: str | Path,
    ) -> list[PDFLayoutRegion]:
        result = self._converter.convert(file_path)
        document = result.document

        regions: list[PDFLayoutRegion] = []

        for item, _ in document.iterate_items():
            prov_list = getattr(item, "prov", None)
            label = getattr(item, "label", None)

            if not prov_list or label is None:
                continue

            text = getattr(item, "text", None)

            for prov in prov_list:
                page = document.pages[prov.page_no]

                bbox = prov.bbox.to_top_left_origin(page_height=page.size.height)

                regions.append(
                    PDFLayoutRegion(
                        label=label.value,
                        page_number=prov.page_no,
                        bbox=BoundingBox(
                            x1=bbox.l,
                            y1=bbox.t,
                            x2=bbox.r,
                            y2=bbox.b,
                        ),
                        text=text,
                    )
                )

        return regions

    def _apply_ocr_fallback(
        self,
        file_path: str | Path,
        regions: list[PDFLayoutRegion],
    ) -> None:
        with fitz.open(file_path) as document:
            for region in regions:
                if not self._needs_ocr(region):
                    continue

                region.text = self._run_ocr_for_region(
                    document,
                    region,
                )

    @staticmethod
    def _needs_ocr(
        region: PDFLayoutRegion,
    ) -> bool:
        if region.label not in OCR_TEXT_LABELS:
            return False

        return region.text is None or not region.text.strip()

    def _run_ocr_for_region(
        self,
        document: fitz.Document,
        region: PDFLayoutRegion,
    ) -> str:
        page_index = region.page_number - 1

        if page_index < 0 or page_index >= len(document):
            return ""

        page = document[page_index]

        clip = fitz.Rect(
            region.bbox.x1,
            region.bbox.y1,
            region.bbox.x2,
            region.bbox.y2,
        )

        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(2.0, 2.0),
            clip=clip,
            alpha=False,
        )

        image = np.frombuffer(
            pixmap.samples,
            dtype=np.uint8,
        ).reshape(
            pixmap.height,
            pixmap.width,
            pixmap.n,
        )

        result = cast(
            RapidOCROutput,
            self._ocr(
                image,
                use_det=True,
                use_cls=False,
                use_rec=True,
            ),
        )

        if result.txts is None:
            return ""

        texts = [text.strip() for text in result.txts if text and text.strip()]

        return "\n".join(texts)

    def _build_pages(
        self,
        regions: list[PDFLayoutRegion],
    ) -> list[Page]:
        pages_by_number: dict[int, list[Block]] = {}

        for order, region in enumerate(regions, start=1):
            block = self._create_block_from_region(
                region,
                order,
            )

            if block is None:
                continue

            pages_by_number.setdefault(
                region.page_number,
                [],
            ).append(block)

        pages: list[Page] = []

        for page_number in sorted(pages_by_number):
            pages.append(
                Page(
                    page_number=page_number,
                    width=None,
                    height=None,
                    blocks=pages_by_number[page_number],
                )
            )

        return pages

    def _create_block_from_region(
        self,
        region: PDFLayoutRegion,
        order: int,
    ) -> Block | None:
        block_type = DOCLING_BLOCK_TYPE_MAP.get(region.label)

        if block_type is None:
            return None

        return Block(
            id=f"block_{uuid4().hex}",
            type=block_type,
            text=region.text or "",
            order=order,
            bbox=region.bbox,
            structure=StructureInfo(
                confidence=1.0,
                source=StructureSource.PARSER,
            ),
        )
