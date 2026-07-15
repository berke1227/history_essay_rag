from tdrag.chunker import chunk_text


def test_bos_metin_bos_liste_dondurur():
    assert chunk_text("", "dosya.pdf", chunk_size_words=10, overlap_words=2) == []


def test_kisa_metin_tek_parca_uretir():
    metin = "bir iki üç dört beş"
    parcalar = chunk_text(metin, "dosya.pdf", chunk_size_words=10, overlap_words=2)
    assert len(parcalar) == 1
    assert parcalar[0].text == metin
    assert parcalar[0].source_file == "dosya.pdf"
    assert parcalar[0].chunk_index == 0


def test_uzun_metin_birden_fazla_parcaya_bolunur():
    kelimeler = [f"kelime{i}" for i in range(50)]
    metin = " ".join(kelimeler)
    parcalar = chunk_text(metin, "dosya.pdf", chunk_size_words=20, overlap_words=5)
    assert len(parcalar) > 1
    # her parça en fazla chunk_size_words kelime içerir
    assert all(len(p.text.split()) <= 20 for p in parcalar)


def test_ortusme_bitisik_parcalar_arasinda_korunur():
    kelimeler = [f"k{i}" for i in range(30)]
    metin = " ".join(kelimeler)
    parcalar = chunk_text(metin, "dosya.pdf", chunk_size_words=10, overlap_words=3)

    ilk_parca_kelimeleri = parcalar[0].text.split()
    ikinci_parca_kelimeleri = parcalar[1].text.split()
    # ilk parçanın son 3 kelimesi, ikinci parçanın ilk 3 kelimesiyle aynı olmalı
    assert ilk_parca_kelimeleri[-3:] == ikinci_parca_kelimeleri[:3]


def test_chunk_id_benzersiz_ve_dosya_adini_icerir():
    metin = " ".join(f"k{i}" for i in range(30))
    parcalar = chunk_text(metin, "makale_a.pdf", chunk_size_words=10, overlap_words=2)
    idler = [p.id for p in parcalar]
    assert len(idler) == len(set(idler))
    assert all("makale_a.pdf" in pid for pid in idler)


def test_chunk_index_sirali_artar():
    metin = " ".join(f"k{i}" for i in range(30))
    parcalar = chunk_text(metin, "dosya.pdf", chunk_size_words=10, overlap_words=2)
    assert [p.chunk_index for p in parcalar] == list(range(len(parcalar)))
