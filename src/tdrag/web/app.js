/**
 * TDRAG Web UI - Ana İstemci Mantığı
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elemanları
  const chatForm = document.getElementById('chatForm');
  const chatInput = document.getElementById('chatInput');
  const sendBtn = document.getElementById('sendBtn');
  const chatContainer = document.getElementById('chatContainer');
  const messagesStream = document.getElementById('messagesStream');
  const welcomeScreen = document.getElementById('welcomeScreen');
  const typingIndicator = document.getElementById('typingIndicator');
  const thinkingStepText = document.getElementById('thinkingStepText');
  const clearChatBtn = document.getElementById('clearChatBtn');

  // Sidebar Elemanları
  const sidebar = document.getElementById('sidebar');
  const openSidebarBtn = document.getElementById('openSidebarBtn');
  const closeSidebarBtn = document.getElementById('closeSidebarBtn');
  const uploadDropzone = document.getElementById('uploadDropzone');
  const pdfFileInput = document.getElementById('pdfFileInput');
  const browseFilesBtn = document.getElementById('browseFilesBtn');
  const uploadProgress = document.getElementById('uploadProgress');
  const uploadProgressBar = document.getElementById('uploadProgressBar');
  const uploadProgressText = document.getElementById('uploadProgressText');
  const articleList = document.getElementById('articleList');
  const loadingArticles = document.getElementById('loadingArticles');
  const articleSearchInput = document.getElementById('articleSearchInput');
  const reindexBtn = document.getElementById('reindexBtn');
  const articleCountSpan = document.getElementById('articleCount');
  const libraryToggleBtn = document.getElementById('libraryToggleBtn');
  const libraryCollapsibleContent = document.getElementById('libraryCollapsibleContent');

  // Hero & Kartal İniş Elemanları
  const eagleLandingHero = document.getElementById('eagleLandingHero');
  const startExploreBtn = document.getElementById('startExploreBtn');
  const replayHeroBtn = document.getElementById('replayHeroBtn');
  const appLayout = document.getElementById('appLayout');

  // Durum Elemanları
  const statModel = document.getElementById('statModel');
  const statNumCtx = document.getElementById('statNumCtx');
  const statChunks = document.getElementById('statChunks');
  const toastContainer = document.getElementById('toastContainer');

  // Odak Makale Elemanları
  const focusBar = document.getElementById('focusBar');
  const focusFilename = document.getElementById('focusFilename');
  const clearFocusBtn = document.getElementById('clearFocusBtn');
  const inputFootnoteText = document.getElementById('inputFootnoteText');

  let allArticles = [];
  let isGenerating = false;
  let selectedArticle = null;
  let isLandingAnimating = false;

  // --------------------------------------------------------------------------
  // BAŞLANGIÇ YÜKLEMESİ
  // --------------------------------------------------------------------------
  fetchStats();
  fetchArticles();

  // --------------------------------------------------------------------------
  // SOHBET VE SORU-CEVAP İŞLEMLERİ
  // --------------------------------------------------------------------------
  chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    handleSendQuestion();
  });

  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendQuestion();
    }
  });

  // Otomatik Büyüyen Textarea
  chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 150) + 'px';
  });

  // Örnek Soru Kartlarına Tıklama
  document.querySelectorAll('.quick-prompt-card').forEach((card) => {
    card.addEventListener('click', () => {
      const promptText = card.getAttribute('data-prompt');
      if (promptText) {
        chatInput.value = promptText;
        handleSendQuestion();
      }
    });
  });

  // Sohbeti Temizle
  clearChatBtn.addEventListener('click', () => {
    messagesStream.innerHTML = '';
    welcomeScreen.style.display = 'flex';
    showToast('Sohbet geçmişi temizlendi.');
  });

  // Odak Temizle
  if (clearFocusBtn) {
    clearFocusBtn.addEventListener('click', () => {
      setArticleFocus(null);
    });
  }

  function setArticleFocus(filename) {
    selectedArticle = filename;
    if (filename) {
      if (focusFilename) focusFilename.textContent = filename;
      if (focusBar) focusBar.style.display = 'flex';
      if (inputFootnoteText) {
        inputFootnoteText.textContent = `🎯 Odak Makale: "${filename}". Sorular sadece bu makaleden yanıtlanacaktır.`;
      }
      showToast(`"${filename}" makalesine odaklanıldı.`);
    } else {
      if (focusBar) focusBar.style.display = 'none';
      if (inputFootnoteText) {
        inputFootnoteText.textContent = '💡 İpucu: Model kütüphanedeki makalelerde geçen bilgileri kullanır. Bir makaleye tıklayarak ona odaklanabilirsiniz.';
      }
      showToast('Tüm kütüphanede arama moduna dönüldü.');
    }
    document.querySelectorAll('.article-item').forEach((item) => {
      if (item.getAttribute('data-filename') === selectedArticle) {
        item.classList.add('active-focus');
      } else {
        item.classList.remove('active-focus');
      }
    });
  }

  async function handleSendQuestion() {
    const question = chatInput.value.trim();
    if (!question || isGenerating) return;

    // Arayüz Hazırlığı
    welcomeScreen.style.display = 'none';
    appendUserMessage(question);
    chatInput.value = '';
    chatInput.style.height = 'auto';
    setGeneratingState(true);

    // Animasyon Basamakları
    const steps = [
      'Vektör veritabanı taranıyor...',
      'İlgili makale parçaları toplanıyor...',
      'Cevap üretiliyor ve kaynak sadakati doğrulanıyor...',
    ];
    let stepIndex = 0;
    thinkingStepText.textContent = steps[0];
    const stepInterval = setInterval(() => {
      stepIndex = (stepIndex + 1) % steps.length;
      thinkingStepText.textContent = steps[stepIndex];
    }, 3000);

    try {
      const reqBody = { question };
      if (selectedArticle) {
        reqBody.source_file = selectedArticle;
      }

      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reqBody),
      });

      clearInterval(stepInterval);

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Cevap alınırken bir hata oluştu.');
      }

      const data = await res.json();
      appendAssistantMessage(data);
    } catch (err) {
      clearInterval(stepInterval);
      appendErrorMessage(err.message);
      showToast(err.message, 'error');
    } finally {
      setGeneratingState(false);
      scrollToBottom();
    }
  }

  function setGeneratingState(generating) {
    isGenerating = generating;
    sendBtn.disabled = generating;
    typingIndicator.style.display = generating ? 'flex' : 'none';
    if (generating) scrollToBottom();
  }

  function appendUserMessage(text) {
    const row = document.createElement('div');
    row.className = 'message-row user-row';
    row.innerHTML = `
      <div class="user-bubble">${escapeHtml(text)}</div>
      <div class="user-avatar">Siz</div>
    `;
    messagesStream.appendChild(row);
    scrollToBottom();
  }

  function appendAssistantMessage(data) {
    const row = document.createElement('div');
    row.className = 'message-row assistant-row';

    // Kaynak rozetleri
    let sourcesHtml = '';
    if (data.sources && data.sources.length > 0) {
      const pills = data.sources
        .map(
          (s) =>
            `<span class="source-pill"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>${escapeHtml(
              s
            )}</span>`
        )
        .join('');
      sourcesHtml = `
        <div class="sources-container">
          <span class="sources-label">Yararlanılan Kaynaklar</span>
          <div class="sources-pills">${pills}</div>
        </div>
      `;
    }

    const formattedAnswer = renderMarkdown(data.answer);
    const groundedClass = data.grounded ? 'true' : 'false';
    const groundedLabel = data.grounded ? '✓ Kaynağa Doğrulandı' : '⚠️ Doğrulanamadı';

    row.innerHTML = `
      <div class="assistant-avatar">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
      </div>
      <div class="assistant-card">
        <div class="assistant-text">${formattedAnswer}</div>
        ${sourcesHtml}
        <div class="message-meta-footer">
          <div class="meta-left">
            <span class="grounded-badge ${groundedClass}">${groundedLabel}</span>
            <span>•</span>
            <span>Deneme: ${data.attempts}</span>
            <span>•</span>
            <span>${data.duration_s}s</span>
          </div>
          <button class="copy-btn" title="Cevabı Kopyala" onclick="navigator.clipboard.writeText(this.closest('.assistant-card').querySelector('.assistant-text').innerText); this.innerText='Kopyalandı!'; setTimeout(() => this.innerText='Kopyala', 2000);">
            Kopyala
          </button>
        </div>
      </div>
    `;
    messagesStream.appendChild(row);
  }

  function appendErrorMessage(errorMsg) {
    const row = document.createElement('div');
    row.className = 'message-row assistant-row';
    row.innerHTML = `
      <div class="assistant-avatar" style="background: #ef4444;">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      </div>
      <div class="assistant-card" style="border-color: #ef4444;">
        <div class="assistant-text" style="color: #fca5a5;">
          <strong>Hata:</strong> ${escapeHtml(errorMsg)}
        </div>
      </div>
    `;
    messagesStream.appendChild(row);
  }

  function scrollToBottom() {
    setTimeout(() => {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }, 50);
  }

  // --------------------------------------------------------------------------
  // İSTATİSTİK VE KÜTÜPHANE YÖNETİMİ
  // --------------------------------------------------------------------------
  async function fetchStats() {
    try {
      const res = await fetch('/api/stats');
      if (!res.ok) return;
      const data = await res.json();
      statModel.textContent = data.model;
      statNumCtx.textContent = `${Math.round(data.num_ctx / 1024)}K (${data.num_ctx})`;
      statChunks.textContent = `${data.total_chunks} Parça`;
    } catch (e) {
      console.warn('İstatistikler yüklenemedi:', e);
    }
  }

  async function fetchArticles() {
    try {
      loadingArticles.style.display = 'flex';
      const res = await fetch('/api/articles');
      if (!res.ok) return;
      allArticles = await res.json();
      articleCountSpan.textContent = allArticles.length;
      renderArticles(allArticles);
    } catch (e) {
      console.warn('Makaleler yüklenemedi:', e);
    } finally {
      loadingArticles.style.display = 'none';
    }
  }

  function renderArticles(list) {
    articleList.innerHTML = '';
    if (list.length === 0) {
      articleList.innerHTML = '<li class="empty-state-list">Makale bulunamadı.</li>';
      return;
    }

    list.forEach((art) => {
      const li = document.createElement('li');
      li.className = 'article-item' + (selectedArticle === art.filename ? ' active-focus' : '');
      li.setAttribute('data-filename', art.filename);
      li.title = `${art.filename} - Odaklanmak için tıklayın`;
      li.innerHTML = `
        <div class="article-info">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          <span class="article-name">${escapeHtml(art.filename)}</span>
        </div>
        <span class="article-size">${art.size_kb} KB</span>
      `;
      li.addEventListener('click', () => {
        if (selectedArticle === art.filename) {
          setArticleFocus(null);
        } else {
          setArticleFocus(art.filename);
        }
      });
      articleList.appendChild(li);
    });
  }

  // Arama Filtresi
  articleSearchInput.addEventListener('input', (e) => {
    const q = e.target.value.toLowerCase().trim();
    if (!q) {
      renderArticles(allArticles);
      return;
    }
    const filtered = allArticles.filter((a) => a.filename.toLowerCase().includes(q));
    renderArticles(filtered);
  });

  // Yeniden İndeksle Butonu
  reindexBtn.addEventListener('click', async (e) => {
    if (e) e.stopPropagation();
    showToast('Tüm makaleler yeniden indeksleniyor...');
    reindexBtn.style.animation = 'spin 1s infinite linear';
    try {
      const res = await fetch('/api/reindex', { method: 'POST' });
      const data = await res.json();
      showToast('İndeksleme başarıyla tamamlandı!', 'success');
      fetchStats();
      fetchArticles();
    } catch (e) {
      showToast('İndeksleme sırasında hata oluştu.', 'error');
    } finally {
      reindexBtn.style.animation = '';
    }
  });

  // --------------------------------------------------------------------------
  // MAKALE YÜKLEME (DRAG AND DROP & FILE PICKER)
  // --------------------------------------------------------------------------
  browseFilesBtn.addEventListener('click', () => {
    pdfFileInput.click();
  });

  pdfFileInput.addEventListener('change', () => {
    if (pdfFileInput.files.length > 0) {
      handleFilesUpload(pdfFileInput.files);
    }
  });

  uploadDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadDropzone.classList.add('drag-active');
  });

  uploadDropzone.addEventListener('dragleave', () => {
    uploadDropzone.classList.remove('drag-active');
  });

  uploadDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadDropzone.classList.remove('drag-active');
    if (e.dataTransfer.files.length > 0) {
      handleFilesUpload(e.dataTransfer.files);
    }
  });

  async function handleFilesUpload(fileList) {
    const pdfFiles = Array.from(fileList).filter((f) => f.name.toLowerCase().endsWith('.pdf'));
    if (pdfFiles.length === 0) {
      showToast('Lütfen yalnızca .pdf uzantılı dosyalar seçin.', 'error');
      return;
    }

    const formData = new FormData();
    pdfFiles.forEach((file) => formData.append('files', file));

    uploadProgress.style.display = 'flex';
    uploadProgressBar.style.width = '40%';
    uploadProgressText.textContent = `${pdfFiles.length} dosya yükleniyor ve indeksleniyor...`;

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      uploadProgressBar.style.width = '100%';

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Yükleme başarısız oldu.');
      }

      const data = await res.json();
      showToast(data.message || 'Makaleler yüklendi ve indekslendi!', 'success');
      pdfFileInput.value = '';
      fetchStats();
      await fetchArticles();
      if (data.saved_files && data.saved_files.length > 0) {
        setArticleFocus(data.saved_files[0]);
      }
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      setTimeout(() => {
        uploadProgress.style.display = 'none';
        uploadProgressBar.style.width = '0%';
      }, 1500);
    }
  }

  // --------------------------------------------------------------------------
  // YARDIMCI İŞLEVLER (HELPERS)
  // --------------------------------------------------------------------------
  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function renderMarkdown(md) {
    if (!md) return '';
    let html = escapeHtml(md);

    // Kalın metin: **metin**
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Eğik metin: *metin*
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // Paragraflar ve yeni satırlar
    const paragraphs = html.split(/\n\n+/);
    html = paragraphs.map((p) => `<p>${p.replace(/\n/g, '<br>')}</p>`).join('');

    return html;
  }

  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(8px)';
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }

  // Mobil Sidebar Açma/Kapama
  if (openSidebarBtn && closeSidebarBtn) {
    openSidebarBtn.addEventListener('click', () => sidebar.classList.add('open'));
    closeSidebarBtn.addEventListener('click', () => sidebar.classList.remove('open'));
  }

  // --------------------------------------------------------------------------
  // MAKALE KÜTÜPHANESİ KATLANABİLİR MENÜ (COLLAPSIBLE ACCORDION)
  // --------------------------------------------------------------------------
  if (libraryToggleBtn && libraryCollapsibleContent) {
    libraryToggleBtn.addEventListener('click', (e) => {
      // Eğer tıklama reindexBtn içinden geldiyse aç/kapa yapma
      if (e.target.closest('#reindexBtn')) return;

      const isCollapsed = libraryCollapsibleContent.classList.toggle('collapsed');
      libraryToggleBtn.classList.toggle('collapsed', isCollapsed);
      try {
        localStorage.setItem('historia_library_collapsed', isCollapsed ? '1' : '0');
      } catch (_) {}
    });

    // Kullanıcının önceki tercihini hatırla
    try {
      if (localStorage.getItem('historia_library_collapsed') === '1') {
        libraryCollapsibleContent.classList.add('collapsed');
        libraryToggleBtn.classList.add('collapsed');
      }
    } catch (_) {}
  }

  // --------------------------------------------------------------------------
  // İNTERAKTİF KARTAL İNİŞİ VE KARŞILAMA SAHNESİ (EAGLE LANDING HERO)
  // --------------------------------------------------------------------------
  function triggerEagleLanding() {
    if (isLandingAnimating || !eagleLandingHero) return;
    isLandingAnimating = true;

    // 1. Kartal gökten süzülüp 2. Friedrich'in koluna doğru inişe geçer
    // (sky katmanı landed katmanına cross-fade yapar, modal zarifçe kaybolur)
    eagleLandingHero.classList.add('landing-active');

    // 2. Kartalın kol üzerine konma ve durulma süresi (~1.25 saniye)
    setTimeout(() => {
      eagleLandingHero.classList.add('landing-finished');
      if (appLayout) {
        appLayout.classList.add('app-ready');
      }
      isLandingAnimating = false;

      // Kullanıcının hemen soru yazabilmesi için girdi alanını odakla
      setTimeout(() => {
        if (chatInput) chatInput.focus();
      }, 350);
    }, 1250);
  }

  function replayEagleHero() {
    if (isLandingAnimating || !eagleLandingHero) return;
    eagleLandingHero.classList.remove('landing-finished', 'landing-active');
    if (appLayout) {
      appLayout.classList.remove('app-ready');
    }
  }

  if (startExploreBtn) {
    startExploreBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      triggerEagleLanding();
    });
  }

  if (eagleLandingHero) {
    eagleLandingHero.addEventListener('click', (e) => {
      // Hero alanında herhangi bir yere tıklandığında iniş başlasın
      triggerEagleLanding();
    });
  }

  if (replayHeroBtn) {
    replayHeroBtn.addEventListener('click', () => {
      replayEagleHero();
    });
  }
});
