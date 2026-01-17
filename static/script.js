const dropZone = document.getElementById('drop-zone');
const videoInput = document.getElementById('video-input');
const browseBtn = document.getElementById('browse-btn');
const progressContainer = document.getElementById('progress-container');
const uploadProgress = document.getElementById('upload-progress');
const statusText = document.getElementById('status-text');
const resultsSection = document.getElementById('results-section');
const reelsGrid = document.getElementById('reels-grid');

const driveLink = document.getElementById('drive-link');
const processLinkBtn = document.getElementById('process-link-btn');

browseBtn.addEventListener('click', () => videoInput.click());

processLinkBtn.addEventListener('click', () => {
    const link = driveLink.value.trim();
    if (link) {
        handleLinkProcess(link);
    } else {
        alert('Please enter a valid Google Drive link.');
    }
});

async function handleLinkProcess(link) {
    dropZone.style.display = 'none';
    progressContainer.style.display = 'block';
    statusText.innerText = 'Processing link...';

    try {
        const response = await fetch('/process-link', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ link: link })
        });

        const data = await response.json();
        if (data.job_id) {
            pollStatus(data.job_id);
        } else {
            throw new Error(data.error || 'Failed to process link');
        }
    } catch (error) {
        console.error('Link processing failed:', error);
        statusText.innerText = 'Link processing failed: ' + error.message;
        setTimeout(() => {
            dropZone.style.display = 'block';
            progressContainer.style.display = 'none';
        }, 3000);
    }
}


dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleUpload(files[0]);
    }
});

videoInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleUpload(e.target.files[0]);
    }
});

async function handleUpload(file) {
    const formData = new FormData();
    formData.append('file', file);

    dropZone.style.display = 'none';
    progressContainer.style.display = 'block';
    statusText.innerText = 'Uploading video...';

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        const jobId = data.job_id;
        pollStatus(jobId);
    } catch (error) {
        console.error('Upload failed:', error);
        statusText.innerText = 'Upload failed. Please try again.';
    }
}

async function pollStatus(jobId) {
    const interval = setInterval(async () => {
        try {
            const response = await fetch(`/status/${jobId}`);
            const data = await response.json();

            // Update status text dynamically from backend
            if (data.status && data.status !== 'completed' && data.status !== 'failed') {
                statusText.innerHTML = `<span class="pulse-dot"></span> ${data.status}`;
            }

            if (data.status === 'completed') {
                clearInterval(interval);
                statusText.innerText = 'Success! Your viral reels are ready.';
                displayResults(data.reels);
            } else if (data.status === 'failed') {
                clearInterval(interval);
                statusText.innerText = 'Processing failed: ' + (data.error || 'Unknown error');
            }
        } catch (error) {
            console.error('Polling failed:', error);
        }
    }, 3000); // Poll every 3 seconds for better responsiveness
}

function displayResults(reels) {
    progressContainer.style.display = 'none';
    resultsSection.style.display = 'block';
    reelsGrid.innerHTML = '';

    reels.forEach(reel => {
        const card = document.createElement('div');
        card.className = 'reel-card';
        card.innerHTML = `
            <video controls>
                <source src="${reel.url}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            <div class="reel-info">
                <h4>${reel.hook}</h4>
                <p>${reel.reason}</p>
                <a href="${reel.url}" download role="button" class="outline" style="margin-top: 1rem; width: 100%;">Download Reel</a>
            </div>
        `;
        reelsGrid.appendChild(card);
    });
}
