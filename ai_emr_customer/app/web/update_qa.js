const data = {
    old_q: formData.get('old_q'),
    old_a: formData.get('old_a'),
    new_q: formData.get('new_q'),
    new_a: formData.get('new_a')
  };
  
  const response = await fetch('https://your-backend-api.com/submit', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  });
  