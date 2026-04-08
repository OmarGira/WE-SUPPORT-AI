from google import genai
from google.genai import types
from src.database import HybridRetriever
import base64
import io
import time
from PIL import Image
from config.settings import GOOGLE_API_KEY, FAISS_INDEX_PATH, BM25_INDEX_PATH

class WEAssistant:
    def __init__(self):
        print("Loading Hybrid Knowledge Base...")
        
        # Spinning up our hybrid search (FAISS for semantic similarity + BM25 for keyword matching).
        # This gives us the best of both worlds for document retrieval.
        self.retriever = HybridRetriever()
        self.retriever.load(FAISS_INDEX_PATH, BM25_INDEX_PATH)
        
        # Initializing the new GenAI client. 
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        
        # Using the flash model here because it's crazy fast, cheap, and handles multimodal tasks perfectly.
        self.model_id = "gemini-flash-latest" 
        
        # Keeping a rolling window of the conversation history so the LLM doesn't lose context.
        # Capped at 6 messages to prevent blowing up the context window (and the API bill).
        self.memory = [] 
        self.memory_limit = 6  
        
        # We define the local vision model placeholder here, but we DON'T load it yet.
        # This saves a massive amount of RAM/VRAM on boot.
        self.local_vision_model = None 
        
        print(f" Connected to: {self.model_id} with Native Memory")

    def _get_local_vision(self):
        """
        Lazy loader for the BLIP model. 
        We only import transformers and load the heavy weights into memory if the user 
        actually toggles the 'use_local_vision' feature. Good for keeping the server lightweight.
        """
        if self.local_vision_model is None:
            print("BLIP local Process - Loading weights into memory...")
            from transformers import pipeline
            # Using BLIP base: it's a solid, lightweight model for generating image captions.
            self.local_vision_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        return self.local_vision_model

    def ask(self, query, file_data=None, image_base64=None, use_local_vision=False):
        # 1. Retrieval Phase: Fetch the top 6 most relevant chunks from our vector DB.
        results = self.retriever.search(query, k=6)
        
        # 2. Context Formatting: Stitching the retrieved docs together.
        context = ""
        if results:
            for i, r in enumerate(results):
                title = r['metadata'].get('title', 'WE Service')
                context += f"({title}): {r['text']}\n" 
        else:
            context = "لا يوجد سياق." # Fallback if nothing is found.

        # 3. Memory Formatting: Flattening the chat history into a string.
        history_text = ""
        if self.memory:
            for msg in self.memory:
                history_text += f"{msg['role']}: {msg['content']}\n"

        # 4. Prompt Engineering: The system prompt that controls the persona.
        # I explicitly told it to stop acting like a Wikipedia article and drop the [1] citations.
        prompt = f"""
        أنت مساعد ذكي ومحترف لخدمة عملاء WE. 
        تعليمات الإجابة:
        1. استخرج إجابتك من "المعلومات المتاحة" او ابحث بدقه لاستخراجها
        2. صغ الإجابة بأسلوب طبيعي، سلس، ومباشر.
        3. ❌ ممنوع منعاً باتاً كتابة أرقام المصادر مثل [1] أو [2] داخل إجابتك.
        4. إذا تم إرفاق صورة أو كود، قم بتحليل المحتوى والإجابة بناءً عليه.
        5. تحدث باللغه اللتي تدخل لك في اول المحادثه
        
        سجل المحادثة:\n{history_text}
        المعلومات المتاحة:\n{context}
        ---
        سؤال العميل: {query}
        """
        
        # Setting up the payload payload. We start with the prompt text.
        request_contents = [prompt]
        
        # If the user uploaded a text-based document (like a txt or parsed PDF/HTML), we append it here.
        # Sliced to 1500 chars just as a safety net against massive text dumps.
        if file_data and isinstance(file_data, str):
            request_contents.append(f"\n\n--- [more info] ---\n{file_data[:1500]}")
            
        # 5. Multimodal Image Handling
        if image_base64:
            import base64
            import io
            from PIL import Image
            try:
                # Convert the base64 string from the frontend back into raw bytes, then into a PIL Image.
                img_bytes = base64.b64decode(image_base64)
                img_object = Image.open(io.BytesIO(img_bytes))
                
                if use_local_vision:
                    # Strategy A: Cost-saving mode. We run the image through our local BLIP model,
                    # grab the text caption, and feed THAT text to Gemini. Zero image tokens billed!
                    vision_model = self._get_local_vision()
                    caption = vision_model(img_object)[0]['generated_text']
                    request_contents[0] += f"\n\n[تحليل الصورة المرفقة بواسطة النظام المحلي]: {caption}"
                else:
                    # Strategy B: High-accuracy mode. We pass the actual image object directly to Gemini
                    # so it can use its native vision capabilities.
                    request_contents.append(img_object)
            except Exception as e:
                print(f"Cannot read the photo: {e}") # Failsafe so the whole app doesn't crash on a bad image.

        # 6. LLM Generation with Robustness (Retry Logic)
        response = None
        # Trying up to 3 times because cloud APIs can be flaky or aggressively rate-limit us.
        for attempt in range(3):
            from google.genai import types
            import time
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=request_contents,
                    # Low temperature (0.1) keeps the model grounded. We want factual support, not creative fiction.
                    config=types.GenerateContentConfig(temperature=0.1) 
                )
                break # If it succeeds, break out of the retry loop.
            except Exception as e:
                error_msg = str(e).lower()
                # Catching specific quota/rate-limiting errors (HTTP 429)
                if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                    if attempt < 2:
                        time.sleep(2) # Exponential/fixed backoff: wait 2 seconds and try again.
                        continue
                    else:
                        raise Exception("عذراً، يوجد ضغط كبير، يرجى الانتظار ثواني.")
                else:
                    # If it's a completely different error (e.g., bad API key), crash immediately so we can debug.
                    raise Exception(f"خطأ من API: {e}")

        # Safety check: if we exhausted the loop and still have nothing, something went terribly wrong.
        if not response:
            raise Exception("فشل في الحصول على رد.")

        # 7. Memory Management: Push the new interaction into our rolling memory window.
        self.memory.append({"role": "العميل", "content": query})
        self.memory.append({"role": "المساعد", "content": response.text})
        if len(self.memory) > self.memory_limit:
            # Drop the oldest messages to maintain the limit.
            self.memory = self.memory[-self.memory_limit:]
            
        # 8. Source Deduplication
        # We extract the URLs from the metadata, but we use a Set to ensure we don't send 
        # the frontend 4 identical links if all retrieved chunks came from the same webpage.
        unique_sources = []
        seen_urls = set()
        for r in results:
            url = r['metadata'].get('source', r['metadata'].get('url', ''))
            title = r['metadata'].get('title', 'موقع WE')
            if url and url not in seen_urls:
                unique_sources.append({"title": title, "url": url})
                seen_urls.add(url)
                
        # Return the final payload: the clean text answer + the unique clickable links.
        return {"answer": response.text, "sources": unique_sources}