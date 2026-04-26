import axios from 'axios';

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const { mode, messages, prompt } = req.body;
  const MODEL_NAME = 'flux-2-klein'; // Defining it here so it matches in both calls

  // --- TEXT MODE: GROQ (Llama 3.1 8B Instant) ---
  if (mode === 'chat') {
    try {
      const response = await axios.post(
        'https://api.groq.com/openai/v1/chat/completions',
        {
          model: 'llama-3.1-8b-instant', 
          messages,
        },
        {
          headers: {
            'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
            'Content-Type': 'application/json',
          },
        }
      );
      return res.status(200).json(response.data);
    } catch (error) {
      console.error('Groq Error:', error.response?.data || error.message);
      return res.status(500).json({ error: 'Groq error: ' + error.message });
    }
  }

  // --- IMAGE MODE: FREEPIK (Fixed Polling) ---
  if (mode === 'image') {
    try {
      // 1. INITIATE TASK
      const initiate = await axios.post(
        `https://api.freepik.com/v1/ai/text-to-image/${MODEL_NAME}`,
        {
          prompt: prompt,
          aspect_ratio: 'square_1_1',
          resolution: '1k'
        },
        {
          headers: {
            'x-freepik-api-key': process.env.FREEPIK_API_KEY,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
        }
      );

      const taskId = initiate.data.data.task_id;
      console.log('Task Created Successfully:', taskId);

      // 2. POLL FOR STATUS (Model-specific URL)
      let imageUrl = null;
      for (let i = 0; i < 15; i++) { // Max 15 attempts
        await new Promise(resolve => setTimeout(resolve, 1500)); // Wait 1.5s
        
        // FIXED URL: Must include the model name in the path
        const statusCheck = await axios.get(
          `https://api.freepik.com/v1/ai/text-to-image/${MODEL_NAME}/${taskId}`, 
          {
            headers: { 
              'x-freepik-api-key': process.env.FREEPIK_API_KEY,
              'Accept': 'application/json'
            }
          }
        );

        const taskData = statusCheck.data.data;
        console.log(`Polling Task (${taskId}):`, taskData.status);

        if (taskData.status === 'COMPLETED') {
          imageUrl = taskData.generated[0]; 
          break;
        }
        
        if (taskData.status === 'ERROR') {
          throw new Error('Freepik AI encountered a generation error.');
        }
      }

      if (!imageUrl) throw new Error('Image generation timed out after polling.');

      return res.status(200).json({ url: imageUrl });

    } catch (error) {
      console.error('Image Error Details:', error.response?.data || error.message);
      return res.status(500).json({ 
        error: error.response?.data?.message || error.message || 'Image generation failed.' 
      });
    }
  }
}