let pc = null;
let mediaRecorder = null;
let recordedChunks = [];

function startWebRTC() {
  pc = new RTCPeerConnection({ sdpSemantics: 'unified-plan' });

  pc.addTransceiver('video', { direction: 'recvonly' });
  pc.addTransceiver('audio', { direction: 'recvonly' });

  pc.ontrack = (event) => {
    if (event.track.kind === 'video') {
      document.getElementById('video').srcObject = event.streams[0];
    } else if (event.track.kind === 'audio') {
      const audioStream = event.streams[0];
      document.getElementById('audio').srcObject = audioStream;

      console.log('🎤 Audio track received, start recording...');
      startRecording(audioStream);
    }
  };

  pc.createOffer()
    .then(offer => pc.setLocalDescription(offer))
    .then(() => waitIceGatheringComplete())
    .then(() => {
      return fetch('/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: pc.localDescription.sdp,
          type: pc.localDescription.type
        })
      });
    })
    .then(response => response.json())
    .then(answer => pc.setRemoteDescription(answer))
    .catch(e => {
      console.error('❌ WebRTC error:', e);
      alert('WebRTC error: ' + e);
    });
}

function waitIceGatheringComplete() {
  return new Promise(resolve => {
    if (pc.iceGatheringState === 'complete') {
      resolve();
    } else {
      const check = () => {
        if (pc.iceGatheringState === 'complete') {
          pc.removeEventListener('icegatheringstatechange', check);
          resolve();
        }
      };
      pc.addEventListener('icegatheringstatechange', check);
    }
  });
}

function startRecording(stream) {
  // 判断浏览器支持的 mimeType
  let mimeType = '';
  if (MediaRecorder.isTypeSupported('audio/webm')) {
    mimeType = 'audio/webm';
  } else if (MediaRecorder.isTypeSupported('audio/ogg')) {
    mimeType = 'audio/ogg';
  } else if (MediaRecorder.isTypeSupported('audio/wav')) {
    mimeType = 'audio/wav';
  } else {
    alert('❌ 不支持的音频编码格式，请使用支持 MediaRecorder 的浏览器（如 Chrome）');
    return;
  }

  try {
    mediaRecorder = new MediaRecorder(stream, {mimeType});
  } catch (e) {
    console.error('🎤 MediaRecorder init failed:', e);
    alert('Recording not supported in this browser.');
    return;
  }

  recordedChunks = [];

  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) {
      recordedChunks.push(event.data);
    }
  };

  mediaRecorder.onstop = async () => {
    console.log('🛑 Recording stopped. Converting to WAV...');
    const webmBlob = new Blob(recordedChunks, { type: 'audio/webm' });

    try {
      const wavBlob = await convertWebMToWav(webmBlob);
      const url = URL.createObjectURL(wavBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'recorded_audio.wav';
      a.style.display = 'none';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      console.log('✅ Audio downloaded.');
    } catch (e) {
      console.error('❌ WAV conversion failed:', e);
    }
  };

  mediaRecorder.start();
  console.log('⏺️ Recording started for 10 seconds...');
  setTimeout(() => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
    }
  }, 10000); // 10秒录音
}

async function convertWebMToWav(webmBlob) {
  const arrayBuffer = await webmBlob.arrayBuffer();
  const audioCtx = new AudioContext();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  const wavBuffer = audioBufferToWav(audioBuffer);
  return new Blob([wavBuffer], { type: 'audio/wav' });
}

function audioBufferToWav(buffer) {
  const numOfChan = buffer.numberOfChannels;
  const length = buffer.length * numOfChan * 2 + 44;
  const view = new DataView(new ArrayBuffer(length));
  let offset = 0;

  function writeStr(s) {
    for (let i = 0; i < s.length; i++) {
      view.setUint8(offset++, s.charCodeAt(i));
    }
  }

  writeStr('RIFF');
  view.setUint32(offset, length - 8, true); offset += 4;
  writeStr('WAVE');
  writeStr('fmt ');
  view.setUint32(offset, 16, true); offset += 4;
  view.setUint16(offset, 1, true); offset += 2;
  view.setUint16(offset, numOfChan, true); offset += 2;
  view.setUint32(offset, buffer.sampleRate, true); offset += 4;
  view.setUint32(offset, buffer.sampleRate * numOfChan * 2, true); offset += 4;
  view.setUint16(offset, numOfChan * 2, true); offset += 2;
  view.setUint16(offset, 16, true); offset += 2;
  writeStr('data');
  view.setUint32(offset, buffer.length * numOfChan * 2, true); offset += 4;

  for (let i = 0; i < buffer.length; i++) {
    for (let ch = 0; ch < numOfChan; ch++) {
      let sample = buffer.getChannelData(ch)[i];
      sample = Math.max(-1, Math.min(1, sample));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
      offset += 2;
    }
  }

  return view.buffer;
}
