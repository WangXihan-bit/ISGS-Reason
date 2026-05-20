import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import re
import base64
import json
import torch
from openai import OpenAI
import numpy as np
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from VLM_plus_detect.inference import inference_vlm
from VLM_plus_detect.llava.model.builder import load_pretrained_model

model_path = "submodules/Qwen3/"
# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_path)
qwen_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="cuda:0"
)

def parse_relations(args, query: str):
    prompt = f"""
            You are an assistant for 3D visual grounding tasks (NR3D/SR3D).
            Your task is to parse a natural language query {query} into structured reasoning steps.

            Rules:
            1. Extract the query into subject–relation–object triples: (subject, relation, object).
            2. Classify each relation as either:
              - "geometry": ONLY if (a) the query contains exactly ONE triple 
                                    (b)the relation is "closest, closer to, farthest, further from", which can be solved purely by Euclidean distance between 3D coordinates.
              - "complex": requires rendering image and visual reasoning (color, texture, material, orientation, appearance, direction, multiple spatial relation).
            3. For each triple, output the reasoning method:
              - If geometry: describe which geometric property to compute (e.g., "find the nearest chair to the table by Euclidean distance").
              - If complex: describe which visual property to extract (e.g., "identify the red color of the chair from rendered image").
            4. All triples MUST share the same subject category. The subject represents the main referred object in the query; other entities appear only as objects in relations.
            5. If the query describes a viewpoint or facing-direction (e.g., contains words like "facing", "looking at", "view from", "in front of camera"),
                then set:
                subject = "" (empty string)
                object = the object being faced, viewed, or referenced by the camera.
            6. Validity Constraint:
              - Not all triples are allowed to have empty subject or object.
              - At least one of (subject, object) must be non-empty in every triple.
              - Reject or reparse any triple where subject == "" and object == "".
            7. When parsing the relation:
              - If the relation phrase contains both a verb and a noun phrase (e.g., "has white curtains", "contains red chairs"), 
                then split it:
                  relation = the main verb or preposition (e.g., "has", "contains")
                  object = the remaining noun phrase (e.g., "white curtains", "red chairs").
              - Relations should typically be verbs or prepositions (e.g., "on", "in", "next to", "behind", "has", "contains", "made of").
              - Avoid leaving the object empty when there is a noun phrase following the relation.

            Output format (strict JSON):
            {{"query_type": "geometry" | "complex",
              "triples": [
                {{"subject": "string",
                  "relation": "string",
                  "object": "string",
                  "type": "geometry" | "complex",
                  "reasoning": "string"
                }}
                ]
            }}
              
            For example:
            Input query : "the closest chair to the table"
            Output: {{ "query_type": "geometry",
              "triples": [
                {{ "subject": "chair",
                  "relation": "closest to",
                  "object": "table",
                  "type": "geometry",
                  "reasoning": "compute Euclidean distances from all chairs to the table and select the minimum"
                }}
                ]
            }}.

            Input query : "facing the window, the chair next to the sofa"
            Output: {{"query_type": "complex",
              "triples": [
                {{ "subject": "",
                  "relation": "facing",
                  "object": "window",
                  "type": "complex",
                  "reasoning": "detection observation direction"
                }},
                {{ "subject": "chair",
                  "relation": "next to",
                  "object": "sofa",
                  "type": "complex",
                  "reasoning": "verify spatial adjacency using rendered image context"
                }}
                ]
            }}.
              
            """

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(args.device)

    generated_ids = qwen_model.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=False
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    # 尝试解析 JSON
    try:
        return json.loads(content)
    except:
        print("Raw model output:", content)
        return None
    
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def inference_with_api(image_path, prompt, model_id="qwen3-vl-plus"):
    """API-based inference using custom endpoint"""
    base64_image = encode_image(image_path)
    client = OpenAI(
        api_key= os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    image_format = image_path.split(".")[-1].lower()
    if image_format == 'jpg':
        image_format = 'jpeg'
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    # Pass in BASE64 image data. Note that the image format (i.e., image/{format}) must match the Content Type in the list of supported images. "f" is the method for string formatting.
                    # PNG image:  f"data:image/png;base64,{base64_image}"
                    # JPEG image: f"data:image/jpeg;base64,{base64_image}"
                    # WEBP image: f"data:image/webp;base64,{base64_image}"
                    "image_url": {"url": f"data:image/{image_format};base64,{base64_image}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    completion = client.chat.completions.create(
        model = model_id,
        messages = messages,
        temperature=0.1,
       
    )
    return completion.choices[0].message.content

def query_images(triples_json, image_path):
    
    prompt = f'''You are given a structured query with triples {triples_json} (subject, relation, object, type, reasoning) and a 3DGS rendered image of a 3D scene.
              

              Your task is to:

              1. Check if the subject and object in the query are both present in the image (estimate degree of presence between 0 and 1).
              2. If present, check whether the described relation holds between them (estimate relation_match between 0 and 1).
              3. Compute and output a single matching score between 0 and 1 (higher means better match) that quantifies how well the image satisfies the query. 
              Use the scoring guideline below to compute the score.
              4. Pay special attention to the subject completeness. A view is preferred when the subject is shown sufficiently completely and clearly. 
              Views where only a small or incomplete part of the subject is visible should receive a low score, even if the relation seems roughly correct.
              Output format:
              {{
              "score": <float 0.00-1.00>,
              "triples": [
              {{
              "subject": "<object A>",
              "relation": "<relation>",
              "object": "<object B>",
              "type": "complex",
              "score": <float 0.00-1.00>
              }}
              ] | [],
              }}

              Scoring guideline (mandatory):

              * Compute the final score as:
                score = 0.4 * subject_presence + 0.3 * object_presence + 0.3 * relation_match
              * Each of subject_presence, object_presence, relation_match must be a number in [0,1].
              * Suggested interpretation:

                * presence = 1.0 if the object is complete and unambiguous.
                * presence = 0.5 if the object is partially visible, occluded, ambiguous, or low-confidence.
                * presence = 0.0 if the object is absolutely absent.
                * relation_match = 1.0 if the described relation clearly and unambiguously holds.
                * relation_match = 0.5 if the relation is plausible/partially holds or ambiguous.
                * relation_match = 0.0 if the relation clearly does not hold.
              * If either subject_presence or object_presence is (close to) 0, set relation_match = 0.
              * Round all reported scores to two decimal places when outputting.

              Example  (partial / ambiguous):
              Query: "The book is on the table."
              Image: book and table present, but book partly occluded and its placement is ambiguous.

              Calculation example:
              subject_presence=0.70, object_presence=1.00, relation_match=0.60
              score = 0.4*0.70 + 0.3*1.00 + 0.3*0.60 = 0.28 + 0.3 + 0.18 = 0.76

              Output:
              {{
              "score": 0.76,
              "triples": [
              {{
              "subject": "book",
              "relation": "on",`0=]
              "object": "table",
              "type": "complex",
              "score": 0.76
              }}
              ],
              }}

              '''
    best_score = -1.0
    best_image_id = None
    
    for image_id in os.listdir(image_path):
      print(image_id)
      image_pth = os.path.join(image_path, image_id)
      response = inference_with_api(image_pth, prompt)
      
      if isinstance(response, str):
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                response = json.loads(match.group(0))
            except json.JSONDecodeError:
                print(f"Failed to parse JSON for {image_id}: {response}")
                response = {}
        else:
            response = {}
      print(response)

      score = response.get("score", 0.0)
      if score > best_score:
        best_score = score
        best_image_id = image_id

    return best_score, best_image_id

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def query_masks(query, image_path, id_list):
    
     # 确保 id_list 是普通 Python int，避免 tensor / numpy int 影响字符串和后处理
    id_list = [int(x) for x in id_list]
    valid_ids_str = ", ".join(map(str, id_list))

    prompt = f'''You are given an image in which each segmented region is marked with a visible numeric ID.

The ONLY valid mask IDs are:
[{valid_ids_str}]

You MUST select mask_id from this given valid ID list.
You MUST NEVER output an ID outside this list, even if you see other numeric IDs in the image.
You MUST NEVER invent a new ID.

Your task is to find the single mask ID from the valid ID list that best matches the SUBJECT in the following query:
{query}

Instructions:
1. Only consider candidate masks whose IDs are in the valid ID list:
   [{valid_ids_str}]
2. Parse the triple(s) carefully and identify:
   - the subject
   - the spatial relation
   - the object
3. Focus primarily on the spatial relation in the triple(s), such as:
   - left of / right of
   - in front of / behind
   - next to / near
   - on / under / above / below
   - inside / between / attached to
4. Use the object as a reference anchor first, then select the subject region whose position best satisfies the described spatial relation with respect to that object.
5. Do not choose a region only because its category looks similar. The selected mask must match both:
   - the subject semantics
   - the spatial relation to the object
6. If multiple candidate masks have similar semantics, choose the one whose spatial position best matches the triple(s).
7. If the object is missing or unclear, still choose the best subject candidate only from the valid ID list, but lower the confidence.
8. If no candidate perfectly matches the query, choose the closest matching ID from the valid ID list.
9. Output only in the following format, with no extra text:

mask_id: X
confidence: Y%

Where:
- X must be one of [{valid_ids_str}]
- Y is an integer from 0 to 100

Important:
- Your main goal is to select the subject mask that best satisfies the spatial relation in the triple(s), not merely the most visually similar object.
- Prefer geometric/spatial consistency over weak semantic similarity.
- If there are multiple triples, choose the mask that best satisfies all triples jointly.
- The final mask_id must be selected strictly from the valid ID list.
'''

    response = inference_with_api(image_path, prompt)
  
    match = re.search(r'mask_id:\s*(\d+)', response)
    if match:
        best_mask_id = int(match.group(1))
        print(best_mask_id) 

    return best_mask_id

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
