"""Boru hattını uçtan uca test edebilmek için 3 kısa örnek PDF makale üretir.

NOT: Bunlar gerçek akademik makaleler DEĞİLDİR — yalnızca PDF çıkarma,
chunking, embedding ve retrieval mekaniğini doğrulamak için yazılmış kısa,
temel düzeyde ve tartışmasız tarihi bilgiler içeren sentetik test verisidir.
Kullanıcı, kendi ~13 makalesini `makaleler/` klasörüne koyduğunda bu script
gerekmez.
"""
from __future__ import annotations

from pathlib import Path

from fpdf import FPDF


def _bul_ttf_font() -> str | None:
    """İşletim sistemine uygun, Türkçe karakter destekleyen bir TTF font arar."""
    olasi_yollar = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    for yol in olasi_yollar:
        if yol.exists():
            return str(yol)
    return None


MAKALELER = {
    "goktürk_kaganligi.pdf": (
        "Göktürk Kağanlığı",
        """Göktürk Kağanlığı, 552 yılında Bumin Kağan önderliğinde Orta Asya'da
kurulmuş bir Türk devletidir. Avarların egemenliğine son verilerek
kurulan kağanlık, kısa sürede Moğolistan'dan Karadeniz'in kuzeyine
kadar uzanan geniş bir bozkır coğrafyasına hakim olmuştur.

582 yılında devlet Doğu ve Batı Göktürk Kağanlığı olmak üzere ikiye
ayrılmıştır. Doğu Göktürk Kağanlığı 630 yılında Tang Hanedanlığı'na
bağlanmış, 682 yılında İlteriş Kutlug Kağan tarafından yeniden
bağımsızlığına kavuşturulmuştur (İkinci Göktürk Kağanlığı).

Orhun Yazıtları, Bilge Kağan, Kül Tigin ve Vezir Tonyukuk adına
dikilmiş olup Türk dilinin bilinen ilk yazılı anıtları arasında kabul
edilir. Yazıtlarda dönemin siyasi olayları, devlet yönetimi anlayışı ve
komşu topluluklarla ilişkiler anlatılmaktadır. Göktürk Kağanlığı 744
yılında Uygurlar tarafından yıkılmıştır.""",
    ),
    "osmanli_devletinin_kurulusu.pdf": (
        "Osmanlı Devleti'nin Kuruluşu",
        """Osmanlı Devleti, 1299 yılı civarında Osman Bey tarafından
Anadolu'nun kuzeybatısında, Söğüt ve çevresinde bir uç beyliği olarak
kurulmuştur. Beylik, Bizans İmparatorluğu'nun Anadolu'daki topraklarına
komşu bir konumda yer almaktaydı.

Orhan Bey döneminde 1326 yılında Bursa fethedilerek başkent yapılmış,
devlet Anadolu beylikleri arasında hızla güçlenmiştir. 1362'de Edirne'nin
alınmasıyla Osmanlılar Balkanlar'da kalıcı bir şekilde yerleşmeye
başlamıştır.

1453 yılında II. Mehmed (Fatih Sultan Mehmed) tarafından İstanbul'un
fethedilmesi, Osmanlı Devleti'nin bir imparatorluğa dönüşüm sürecinde
dönüm noktalarından biri kabul edilir. Devlet, 1922 yılına kadar varlığını
sürdürmüştür.""",
    ),
    "buyuk_selcuklu_devleti.pdf": (
        "Büyük Selçuklu Devleti",
        """Büyük Selçuklu Devleti, Oğuzların Kınık boyuna mensup Selçuk Bey'in
soyundan gelen Tuğrul ve Çağrı Bey kardeşler tarafından 1040 yılındaki
Dandanakan Savaşı sonrasında Gazneliler'e karşı kazanılan zaferin
ardından kurulmuştur.

1055 yılında Tuğrul Bey'in Bağdat'a girmesiyle Abbasi halifeliği
üzerindeki Büveyhi etkisi sona ermiş, Tuğrul Bey halife tarafından
'Doğu'nun ve Batı'nın Sultanı' unvanıyla tanınmıştır.

1071 Malazgirt Savaşı'nda Sultan Alparslan komutasındaki Selçuklu
ordusu Bizans İmparatorluğu'nu yenilgiye uğratmış, bu zafer Anadolu'nun
Türkleşme sürecinin başlangıcı olarak kabul edilir. Melikşah döneminde
devlet en geniş sınırlarına ulaşmış, veziri Nizamülmülk döneminde
Nizamiye medreseleri kurulmuştur.""",
    ),
}


def olustur(cikti_klasoru: Path) -> list[Path]:
    cikti_klasoru.mkdir(parents=True, exist_ok=True)
    yollar = []
    font_yolu = _bul_ttf_font()

    for dosya_adi, (baslik, govde) in MAKALELER.items():
        pdf = FPDF()
        pdf.add_page()
        if font_yolu:
            pdf.add_font("CustomUnicode", "", font_yolu)
            pdf.set_font("CustomUnicode", size=16)
            pdf.multi_cell(0, 10, baslik)
            pdf.ln(4)
            pdf.set_font("CustomUnicode", size=11)
            pdf.multi_cell(0, 7, govde.strip())
        else:
            pdf.set_font("Helvetica", size=16)
            pdf.multi_cell(0, 10, baslik.encode("latin-1", errors="replace").decode("latin-1"))
            pdf.ln(4)
            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(0, 7, govde.strip().encode("latin-1", errors="replace").decode("latin-1"))

        yol = cikti_klasoru / dosya_adi
        pdf.output(str(yol))
        yollar.append(yol)
    return yollar


if __name__ == "__main__":
    hedef = Path(__file__).resolve().parent.parent / "makaleler"
    olusturulanlar = olustur(hedef)
    for yol in olusturulanlar:
        print(f"oluşturuldu: {yol}")
