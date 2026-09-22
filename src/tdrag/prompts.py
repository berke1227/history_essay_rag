"""LLM'e gönderilen prompt şablonları. Kurallar promptun içine gömülüdür;
model çıktıyı bu kurallara göre üretir, kod tarafında ayrıca zorlanmaz
(uzunluk/JSON biçimi hariç — onlar qa_pipeline'da doğrulanır)."""

ANSWER_PROMPT = """Sen, yalnızca sağlanan makale alıntılarına dayanarak çalışan bir tarih araştırma asistanısın.

KURALLAR:
1. SADECE aşağıdaki "MAKALE ALINTILARI" bölümünde geçen bilgileri kullan. Dışarıdan bilgi ekleme, tahmin yürütme veya araştırma yapma.
2. Eğer sorunun cevabı alıntılarda yoksa, başka hiçbir şey eklemeden şunu söyle: "Bu konu verilen makalelerde ele alınmamış."
3. Cevap bulunabiliyorsa en az bir paragraf uzunluğunda, açıklayıcı ve akıcı olmalı.
4. Eğer alıntılarda birden fazla farklı makale veya devlet geçiyorsa, SADECE soruda sorulan konu/devlet ile ilgili olan bilgileri kullan. Alıntılardaki diğer alakasız devletlerden veya konulardan ASLA bahsetme.
{geri_bildirim_bolumu}
MAKALE ALINTILARI:
{context}

SORU: {question}

CEVAP:"""

GERI_BILDIRIM_SABLONU = """
ÖNCEKİ DENEMEN REDDEDİLDİ. Şu sorunları düzelterek yeniden yanıtla: {feedback}
"""

VERIFY_PROMPT = """Sen bir cevap denetleyicisisin. Aşağıdaki soruyu, kaynak alıntıları ve üretilen cevabı incele.

SORU: {question}

KAYNAK ALINTILAR:
{context}

ÜRETİLEN CEVAP:
{answer}

Şu üç kriteri kontrol et:
1. Cevap SADECE kaynak alıntılarda yer alan bilgilere dayanıyor mu (alıntılarda hiç bulunmayan tamamen uydurma/dış dünya bilgisi var mı)?
2. Cevap en az bir paragraf uzunluğunda, dolu ve açıklayıcı mı?
3. Eğer soru kaynak alıntılarda geçmiyorsa, cevap bunu doğru şekilde belirtiyor mu ("Bu konu verilen makalelerde ele alınmamış")?

ÖNEMLİ KILAVUZ:
- Eşanlamlı sözcükler, doğal cümle yapıları veya alıntıdaki bilgilerin akıcı şekilde özetlenmesi uydurma sayılmaz.
- Cevabın reddedilmesi için kaynakta hiç olmayan uydurma bir olgu veya açık bir tahrifat bulunmalıdır.
- Soru belirli bir konuya yönelikse (örneğin Normanlar), cevabın sadece o konuyu ele alması doğrudur; alıntılardaki diğer alakasız devletlerin cevapta geçmemesi bir eksiklik değildir.

SADECE aşağıdaki JSON formatında yanıt ver, başka hiçbir metin ekleme:
{{"gecti": true veya false, "geri_bildirim": "sorun varsa en fazla 1-2 cümlelik kısa ve somut açıklama, yoksa boş string"}}"""


def answer_prompt_olustur(question: str, context: str, feedback: str | None) -> str:
    geri_bildirim_bolumu = GERI_BILDIRIM_SABLONU.format(feedback=feedback) if feedback else ""
    return ANSWER_PROMPT.format(
        context=context, question=question, geri_bildirim_bolumu=geri_bildirim_bolumu
    )


def verify_prompt_olustur(question: str, context: str, answer: str) -> str:
    return VERIFY_PROMPT.format(question=question, context=context, answer=answer)
