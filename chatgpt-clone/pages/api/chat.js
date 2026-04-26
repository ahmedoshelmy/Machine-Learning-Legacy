import axios from 'axios';

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const { mode, messages, prompt, model, negativePrompt, aspect_ratio, num_images, guidance_scale } = req.body;

  // ── CHAT MODE (Groq) ────────────────────────────────────────────────────────
  if (mode === 'chat') {
    const chatModel = model || 'llama-3.1-8b-instant';
    try {
      const response = await axios.post(
        'https://api.groq.com/openai/v1/chat/completions',
        { model: chatModel, messages },
        { headers: { 'Authorization': `Bearer ${process.env.GROQ_API_KEY}`, 'Content-Type': 'application/json' } }
      );
      return res.status(200).json(response.data);
    } catch (error) {
      console.error('Groq Error:', error.response?.data || error.message);
      return res.status(500).json({ error: error.response?.data?.error?.message || error.message });
    }
  }

  // ── IMAGE MODE (Freepik) ────────────────────────────────────────────────────
  if (mode === 'image') {
    const imageModel = model || 'flux-2-klein';
    try {
      // 1. Initiate generation
      const body = {
        prompt,
        aspect_ratio: aspect_ratio || 'square_1_1',
        num_images:   num_images   || 1,
        guidance_scale: guidance_scale ?? 1.0,
      };
      if (negativePrompt?.trim()) body.negative_prompt = negativePrompt;

      const initiate = await axios.post(
        `https://api.freepik.com/v1/ai/text-to-image/${imageModel}`,
        body,
        { headers: { 'x-freepik-api-key': process.env.FREEPIK_API_KEY, 'Content-Type': 'application/json', 'Accept': 'application/json' } }
      );

      const taskId = initiate.data.data?.task_id;
      if (!taskId) throw new Error('No task_id returned from Freepik.');
      console.log('Freepik task created:', taskId);

      // 2. Poll for completion
      let imageUrl = null;
      let seed = null;
      for (let i = 0; i < 20; i++) {
        await new Promise(r => setTimeout(r, 1500));
        const poll = await axios.get(
          `https://api.freepik.com/v1/ai/text-to-image/${imageModel}/${taskId}`,
          { headers: { 'x-freepik-api-key': process.env.FREEPIK_API_KEY, 'Accept': 'application/json' } }
        );
        const taskData = poll.data.data;
        console.log(`Poll [${i + 1}] status:`, taskData.status);

        if (taskData.status === 'COMPLETED') {
          imageUrl = taskData.generated?.[0];
          seed     = taskData.seed ?? null;
          break;
        }
        if (taskData.status === 'ERROR') throw new Error('Freepik generation error.');
      }

      if (!imageUrl) throw new Error('Image generation timed out.');
      return res.status(200).json({ url: imageUrl, seed });

    } catch (error) {
      console.error('Image Error:', error.response?.data || error.message);
      return res.status(500).json({ error: error.response?.data?.message || error.message });
    }
  }

  return res.status(400).json({ error: 'Invalid mode.' });
}
