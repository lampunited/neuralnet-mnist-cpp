const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const resultDiv = document.getElementById('result');

let drawing = false;
ctx.lineWidth = 30;  
ctx.lineCap = 'round';
ctx.strokeStyle = '#000';

ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);

canvas.addEventListener('mousedown', () => drawing = true);
canvas.addEventListener('mouseup',   () => {
  drawing = false;
  ctx.beginPath();  
});
canvas.addEventListener('mouseout',  () => {
  drawing = false;
  ctx.beginPath();
});
canvas.addEventListener('mousemove', draw);

function draw(e) {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
}

clearBtn.addEventListener('click', () => {
  ctx.fillStyle = 'white';  
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  resultDiv.textContent = '';
});

predictBtn.addEventListener('click', async () => {
  const small = document.createElement('canvas');
  small.width = 28; small.height = 28;
  const sctx = small.getContext('2d');
  sctx.drawImage(canvas, 0, 0, 28, 28);

  const imgData = sctx.getImageData(0, 0, 28, 28).data;
  const pixels = [];
  for (let i = 0; i < imgData.length; i += 4) {
    const brightness = imgData[i] / 255;
    pixels.push(1 - brightness);  
  }

  console.log("Pixels being sent:", pixels.slice(0, 50));  

  try {
    const resp = await fetch('http://localhost:8080/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pixels })
    });

    const result = await resp.json();
    console.log("Server response:", result);

    const { digit, confidence } = result;
    resultDiv.textContent = `I think it's a ${digit} (${(confidence * 100).toFixed(1)}% confident)`;
  } catch (err) {
    resultDiv.textContent = 'Error: ' + err.message;
    console.error(err);
  }
});
