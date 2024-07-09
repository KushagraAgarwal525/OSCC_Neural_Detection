
// script.js
document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const preview = document.getElementById('preview');
    const result = document.getElementById('result');
    const selectFileButton = document.getElementById('select-file');

    selectFileButton.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragging');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragging');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragging');
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    const handleFiles = (files) => {
        const file = files[0];
        if (file && file.type.startsWith('image/')) {
            previewFile(file);
            uploadFile(file);
        } else {
            alert('Please upload an image file.');
        }
    };

    const previewFile = (file) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = () => {
            const img = document.createElement('img');
            img.src = reader.result;
            preview.innerHTML = '';
            preview.appendChild(img);
        };
    };

    const uploadFile = (file) => {
        
        const formData = new FormData();
        formData.append('file', file);

        axios.post('/predict', formData)
            .then(response => {
                console.log('Image uploaded successfully', response);
                result.textContent = response.data.message;
            })
            .catch(error => {
                console.error('Error uploading image', error);
            });
    };
});
    