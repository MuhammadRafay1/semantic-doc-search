/* ═══════════════════════════════════════════════════════════
   APIMatic Doc Search — Frontend Application Logic
   ═══════════════════════════════════════════════════════════ */

// ─── State ───
const HISTORY_KEY = 'apimatic_doc_search_history';
const MAX_HISTORY = 15;

// ─── Initialization ───
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    loadHistory();

    // Enter key binding
    document.getElementById('searchInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') performSearch();
    });
});

// ─── Health Check ───
async function checkHealth() {
    const dot = document.getElementById('statusDot');
    const text = document.getElementById('statusText');

    try {
        const res = await fetch('/api/health');
        const data = await res.json();

        if (data.index_loaded) {
            dot.className = 'status-dot online';
            const llmStatus = data.llm_available ? 'Index + LLM Ready' : 'Index Ready (No LLM)';
            text.textContent = llmStatus;
        } else {
            dot.className = 'status-dot offline';
            text.textContent = 'Index not loaded';
        }
    } catch {
        dot.className = 'status-dot offline';
        text.textContent = 'Offline';
    }
}

// ─── Search ───
async function performSearch() {
    const query = document.getElementById('searchInput').value.trim();
    if (!query) return;

    const topK = parseInt(document.getElementById('topK').value);
    const useLLM = document.getElementById('useLLM').checked;
    const btn = document.getElementById('searchBtn');
    const resultsContainer = document.getElementById('resultsContainer');

    // UI: Loading state
    btn.classList.add('loading');
    btn.disabled = true;
    resultsContainer.style.display = 'block';
    document.getElementById('suggestions').style.display = 'none';

    // Save to history
    addToHistory(query);

    if (useLLM) {
        await performStreamingSearch(query, topK);
    } else {
        await performQuickSearch(query, topK);
    }

    // UI: Reset button
    btn.classList.remove('loading');
    btn.disabled = false;
}

// ─── Streaming Search (with LLM) ───
async function performStreamingSearch(query, topK) {
    const aiCard = document.getElementById('aiAnswerCard');
    const aiBody = document.getElementById('aiAnswerBody');
    const aiMeta = document.getElementById('aiMeta');
    const sourcesGrid = document.getElementById('sourcesGrid');
    const sourcesHeader = document.getElementById('sourcesHeader');
    const resultCount = document.getElementById('resultCount');
    const llmNotice = document.getElementById('llmNotice');

    // Reset UI
    aiCard.style.display = 'block';
    aiBody.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    aiMeta.textContent = '';
    sourcesGrid.innerHTML = '';
    llmNotice.style.display = 'none';
    sourcesHeader.style.display = 'none';

    const startTime = performance.now();

    try {
        const res = await fetch('/api/search/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, top_k: topK, use_llm: true }),
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let answerText = '';
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;

                try {
                    const event = JSON.parse(line.slice(6));

                    switch (event.type) {
                        case 'sources':
                            renderSources(event.data);
                            sourcesHeader.style.display = 'flex';
                            resultCount.textContent = `${event.data.length} documents`;
                            break;

                        case 'token':
                            if (answerText === '') {
                                aiBody.innerHTML = ''; // Clear typing indicator
                            }
                            answerText += event.data;
                            aiBody.innerHTML = renderMarkdown(answerText);
                            aiBody.scrollTop = aiBody.scrollHeight;
                            break;

                        case 'error':
                            if (answerText === '') {
                                aiCard.style.display = 'none';
                                llmNotice.style.display = 'flex';
                                document.getElementById('llmNoticeText').textContent = event.data;
                            }
                            break;

                        case 'done':
                            const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
                            aiMeta.textContent = `${elapsed}s`;
                            break;
                    }
                } catch {
                    // Skip malformed JSON
                }
            }
        }
    } catch (err) {
        aiCard.style.display = 'none';
        llmNotice.style.display = 'flex';
        document.getElementById('llmNoticeText').textContent = `Search failed: ${err.message}`;

        // Fallback to quick search
        await performQuickSearch(query, topK);
    }
}

// ─── Quick Search (vector only) ───
async function performQuickSearch(query, topK) {
    const aiCard = document.getElementById('aiAnswerCard');
    const sourcesGrid = document.getElementById('sourcesGrid');
    const sourcesHeader = document.getElementById('sourcesHeader');
    const resultCount = document.getElementById('resultCount');
    const llmNotice = document.getElementById('llmNotice');

    aiCard.style.display = 'none';
    sourcesGrid.innerHTML = '';
    llmNotice.style.display = 'none';

    try {
        const res = await fetch(`/api/search/quick?q=${encodeURIComponent(query)}&k=${topK}`);
        const data = await res.json();

        if (data.results && data.results.length > 0) {
            renderSources(data.results);
            sourcesHeader.style.display = 'flex';
            resultCount.textContent = `${data.results.length} documents · ${data.latency_ms.toFixed(0)}ms`;
        } else {
            sourcesGrid.innerHTML = '<div class="no-results">No results found. Try a different query.</div>';
            sourcesHeader.style.display = 'flex';
            resultCount.textContent = '0 documents';
        }
    } catch (err) {
        sourcesGrid.innerHTML = `<div class="no-results">Search failed: ${err.message}</div>`;
        sourcesHeader.style.display = 'flex';
        resultCount.textContent = 'Error';
    }
}

// ─── Render Sources ───
function renderSources(sources) {
    const grid = document.getElementById('sourcesGrid');
    grid.innerHTML = sources.map((src, idx) => {
        const score = src.relevance_score;
        const scoreClass = score < 0.5 ? 'high' : score < 1.0 ? 'medium' : 'low';
        const scoreLabel = score < 0.5 ? 'High' : score < 1.0 ? 'Medium' : 'Low';
        const preview = escapeHtml(src.preview || src.chunk_preview || '');

        return `
            <div class="source-card" style="animation: fadeInUp 0.3s ease ${idx * 0.05}s both;">
                <div class="source-card-header">
                    <div class="source-file">
                        <div class="source-file-icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                                <polyline points="14 2 14 8 20 8"/>
                                <line x1="16" y1="13" x2="8" y2="13"/>
                                <line x1="16" y1="17" x2="8" y2="17"/>
                            </svg>
                        </div>
                        <div>
                            <div class="source-filename">${escapeHtml(src.filename)}</div>
                            ${src.title ? `<div class="source-title">${escapeHtml(src.title)}</div>` : ''}
                        </div>
                    </div>
                    <span class="source-score ${scoreClass}">${scoreLabel} · ${score.toFixed(3)}</span>
                </div>
                ${src.category ? `<span class="source-category">${escapeHtml(src.category)}</span>` : ''}
                <div class="source-preview">${preview}</div>
            </div>
        `;
    }).join('');
}

// ─── Markdown Renderer (lightweight) ───
function renderMarkdown(text) {
    let html = escapeHtml(text);

    // Code blocks (```)
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Italic
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Headers
    html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Unordered lists
    html = html.replace(/^[*-] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');

    // Ordered lists
    html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    // Links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

    // Paragraphs (double newlines)
    html = html.replace(/\n\n/g, '</p><p>');

    // Single newlines to <br>
    html = html.replace(/\n/g, '<br>');

    return `<p>${html}</p>`;
}

// ─── Suggestions ───
function searchSuggestion(el) {
    document.getElementById('searchInput').value = el.textContent;
    performSearch();
}

// ─── History ───
function loadHistory() {
    const history = getHistory();
    const section = document.getElementById('historySection');
    const list = document.getElementById('historyList');

    if (history.length === 0) {
        section.style.display = 'none';
        return;
    }

    section.style.display = 'block';
    list.innerHTML = history.map(q =>
        `<button class="history-item" onclick="searchFromHistory(this)">${escapeHtml(q)}</button>`
    ).join('');
}

function getHistory() {
    try {
        return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
    } catch {
        return [];
    }
}

function addToHistory(query) {
    let history = getHistory();
    history = history.filter(q => q !== query);
    history.unshift(query);
    history = history.slice(0, MAX_HISTORY);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    loadHistory();
}

function clearHistory() {
    localStorage.removeItem(HISTORY_KEY);
    loadHistory();
}

function searchFromHistory(el) {
    document.getElementById('searchInput').value = el.textContent;
    performSearch();
}

// ─── Utils ───
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}