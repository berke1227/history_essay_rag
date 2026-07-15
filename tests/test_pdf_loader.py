from pathlib import Path

import pytest

from tdrag.pdf_loader import PdfMetniCikarilamadiError, extract_text, load_articles


def test_gecerli_pdfden_turkce_metin_dogru_cikarilir(sample_pdf_folder: Path):
    pdf_yolu = sample_pdf_folder / "osmanli_devletinin_kurulusu.pdf"
    metin = extract_text(pdf_yolu)
    assert "Osman Bey" in metin
    assert "Söğüt" in metin  # Türkçe karakter (ğ) doğru çıkarılmalı


def test_load_articles_klasordeki_tum_pdfleri_okur(sample_pdf_folder: Path):
    basarili, basarisiz = load_articles(sample_pdf_folder)
    assert len(basarili) == 3
    assert basarisiz == []
    dosya_adlari = {ad for ad, _ in basarili}
    assert dosya_adlari == {
        "goktürk_kaganligi.pdf",
        "osmanli_devletinin_kurulusu.pdf",
        "buyuk_selcuklu_devleti.pdf",
    }


def test_olmayan_klasor_hata_firlatir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_articles(tmp_path / "olmayan_klasor")


def test_bos_klasor_hata_firlatir(tmp_path: Path):
    bos = tmp_path / "bos"
    bos.mkdir()
    with pytest.raises(FileNotFoundError):
        load_articles(bos)


def test_bozuk_pdf_tek_basina_hata_firlatir(tmp_path: Path):
    sahte_pdf = tmp_path / "bozuk.pdf"
    sahte_pdf.write_text("bu bir pdf değil")
    with pytest.raises(Exception):
        extract_text(sahte_pdf)


def test_bozuk_pdf_toplu_islemi_durdurmaz(sample_pdf_folder: Path):
    (sample_pdf_folder / "bozuk.pdf").write_text("bu geçerli bir pdf değil")

    basarili, basarisiz = load_articles(sample_pdf_folder)

    assert len(basarili) == 3
    assert len(basarisiz) == 1
    assert basarisiz[0][0] == "bozuk.pdf"


def test_metin_katmani_olmayan_pdf_acik_hata_verir(tmp_path: Path):
    """Taranmış görüntü PDF'lerini simüle eder: metinsiz, boş bir sayfa
    içeren gerçek bir PDF üretip extract_text'in bunu sessizce boş
    geçmek yerine açıkça reddettiğini doğrular (OCR desteklenmiyor)."""
    from fpdf import FPDF

    bos_pdf_yolu = tmp_path / "taranmis_gibi_bos.pdf"
    pdf = FPDF()
    pdf.add_page()  # hiç metin eklenmedi -> metin katmanı yok
    pdf.output(str(bos_pdf_yolu))

    with pytest.raises(PdfMetniCikarilamadiError):
        extract_text(bos_pdf_yolu)
