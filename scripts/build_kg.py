import json
import re
import logging
from typing import Dict, List, Tuple, Any
import openai
from openai import OpenAI
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    def __init__(self, openai_client: OpenAI, model: str = "gpt-4-turbo"):
        self.client = openai_client
        self.model = model
        self.base_edges = []
        self.augmented_edges = []
        self.final_edges = []
        
    def clean_json_response(self, text: str) -> str:
        """Clean and extract JSON from LLM response"""
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Find JSON-like content between curly braces
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        # If no JSON found, try to find array
        array_match = re.search(r'\[.*\]', text, re.DOTALL)
        if array_match:
            return array_match.group(0)
        
        return text.strip()
    
    def extract_with_retry(self, prompt: str, max_retries: int = 3) -> Dict:
        """Extract information with retry logic for robust JSON parsing"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=3000
                )
                
                content = response.choices[0].message.content
                logger.info(f"Raw response (attempt {attempt + 1}): {content[:200]}...")
                
                # Clean the response
                clean_content = self.clean_json_response(content)
                
                # Try to parse JSON
                result = json.loads(clean_content)
                return result
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    # Fallback: try to extract information manually
                    return self.fallback_extraction(content)
                time.sleep(1)  # Brief pause before retry
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return {"edges": []}
                time.sleep(1)
        
        return {"edges": []}
    
    def fallback_extraction(self, content: str) -> Dict:
        """Manual extraction when JSON parsing fails"""
        logger.info("Using fallback extraction method")
        
        edges = []
        
        # Try to extract basic relationships from text
        lines = content.split('\n')
        for line in lines:
            if '->' in line or 'relationship:' in line.lower():
                # Basic pattern matching for relationships
                parts = line.split('->')
                if len(parts) == 2:
                    subject = parts[0].strip()
                    object_rel = parts[1].strip()
                    
                    edge = {
                        "subject": subject,
                        "subject_label": subject,
                        "relationship": f"{subject} related to {object_rel}",
                        "object": object_rel,
                        "object_label": object_rel
                    }
                    edges.append(edge)
        
        return {"edges": edges}
    
    def step1_unified_extraction(self, text: str) -> List[Dict]:
        """
        Step 1: Unified Ontology and Relationship Extraction
        Uses one-shot learning to extract semantic relationships as subject-predicate-object triples
        """
        logger.info("Step 1: Performing unified ontology and relationship extraction...")
        
        # One-shot learning example
        example_input = """The Treaty of Versailles was signed in 1919 at the Palace of Versailles. It ended World War I and imposed harsh conditions on Germany."""
        
        example_output = """{
  "edges": [
    {
      "subject": "Treaty of Versailles",
      "subject_label": "Peace treaty that ended World War I",
      "relationship": "Treaty of Versailles was signed in 1919",
      "object": "1919",
      "object_label": "Year when the Treaty of Versailles was signed"
    },
    {
      "subject": "Treaty of Versailles", 
      "subject_label": "Peace treaty that ended World War I",
      "relationship": "Treaty of Versailles was signed at Palace of Versailles",
      "object": "Palace of Versailles",
      "object_label": "Location where the Treaty of Versailles was signed"
    }
  ]
}"""
        
        prompt = f"""You are an expert knowledge graph extraction system. Your task is to extract semantic relationships as subject-predicate-object triples from text.

EXAMPLE:
Input: {example_input}
Output: {example_output}

Now extract relationships from the following text. Follow these guidelines:
1. Extract entities, events, dates, locations, and numerical measurements
2. Create descriptive relationships between them
3. Provide clear, informative labels for subjects and objects
4. Focus on factual, verifiable relationships
5. Return ONLY valid JSON in the exact format shown above

Input text: {text[:2500]}

Output:"""
        
        result = self.extract_with_retry(prompt)
        extracted_edges = result.get("edges", [])
        
        logger.info(f"Step 1 completed: Extracted {len(extracted_edges)} base relationships")
        return extracted_edges
    
    def step2_knowledge_augmentation(self, original_text: str, base_graph: List[Dict]) -> List[Dict]:
        """
        Step 2: Knowledge Augmentation and Refinement
        Provides original text and base graph to identify missing entities, relationships, dates, or measurements
        """
        logger.info("Step 2: Performing knowledge augmentation and refinement...")
        
        # Convert base graph to readable format for the prompt
        base_graph_text = ""
        for i, edge in enumerate(base_graph[:20]):  # Limit to avoid token overflow
            base_graph_text += f"{i+1}. {edge['subject']} --[{edge['relationship']}]--> {edge['object']}\n"
        
        prompt = f"""You are performing knowledge augmentation on an existing knowledge graph. Your task is to identify and add any MISSING entities, relationships, key dates, or numerical measurements that were overlooked in the initial extraction.

ORIGINAL TEXT:
{original_text[:2000]}

EXISTING GRAPH (already extracted):
{base_graph_text}

Your task:
1. Carefully review the original text
2. Identify any important entities, relationships, dates, or measurements NOT captured in the existing graph
3. Extract ONLY the MISSING information as new edges
4. Focus particularly on:
   - Key dates and temporal relationships
   - Numerical measurements and quantities
   - Causal relationships
   - Geographic and organizational connections
   - Missing entities that play important roles

Return ONLY the NEW/MISSING edges in this JSON format:
{{
  "edges": [
    {{
      "subject": "entity_name",
      "subject_label": "detailed description",
      "relationship": "descriptive relationship statement", 
      "object": "related_entity",
      "object_label": "detailed description"
    }}
  ]
}}

Do NOT repeat relationships already in the existing graph. Only extract what is missing."""
        
        result = self.extract_with_retry(prompt)
        augmented_edges = result.get("edges", [])
        
        logger.info(f"Step 2 completed: Found {len(augmented_edges)} additional relationships")
        return augmented_edges
    
    def step3_merge_and_deduplicate(self, base_edges: List[Dict], augmented_edges: List[Dict]) -> List[Dict]:
        """
        Step 3: Merge base and augmented graphs, avoiding duplicates
        """
        logger.info("Step 3: Merging graphs and removing duplicates...")
        
        all_edges = base_edges + augmented_edges
        
        # Deduplicate based on subject, object, and relationship similarity
        seen = set()
        unique_edges = []
        
        for edge in all_edges:
            subject = edge.get("subject", "").lower().strip()
            object_name = edge.get("object", "").lower().strip()
            relationship = edge.get("relationship", "").lower().strip()
            
            # Create a normalized key for deduplication
            key = (subject, object_name, relationship)
            
            if key not in seen and all([subject, object_name, relationship]):
                seen.add(key)
                unique_edges.append(edge)
        
        logger.info(f"Step 3 completed: {len(unique_edges)} unique edges after merging and deduplication")
        return unique_edges
    
    def process_text_with_pipeline(self, text: str, chunk_size: int = 1500):
        """
        Complete pipeline: Extract, Augment, and Merge
        Following the methodology from the paper
        """
        logger.info("Starting complete graph construction pipeline...")
        
        # Split text into manageable chunks
        sections = self._split_text_into_sections(text, chunk_size)
        
        all_base_edges = []
        all_augmented_edges = []
        
        for i, section in enumerate(sections):
            if len(section.strip()) > 100:
                logger.info(f"Processing section {i+1}/{len(sections)}")
                
                # Step 1: Unified extraction
                base_edges = self.step1_unified_extraction(section)
                all_base_edges.extend(base_edges)
                
                # Step 2: Knowledge augmentation
                augmented_edges = self.step2_knowledge_augmentation(section, base_edges)
                all_augmented_edges.extend(augmented_edges)
                
                # Small delay to avoid rate limiting
                time.sleep(1)
        
        # Step 3: Merge and deduplicate
        self.base_edges = all_base_edges
        self.augmented_edges = all_augmented_edges
        self.final_edges = self.step3_merge_and_deduplicate(all_base_edges, all_augmented_edges)
        
        logger.info("Pipeline completed successfully!")
    
    def _split_text_into_sections(self, text: str, chunk_size: int) -> List[str]:
        """Split text into sections for processing"""
        sections = []
        
        # First try to split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        for paragraph in paragraphs:
            if len(current_chunk + paragraph) < chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    sections.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            sections.append(current_chunk.strip())
        
        # If sections are still too large, split by sentences
        if any(len(section) > chunk_size for section in sections):
            new_sections = []
            for section in sections:
                if len(section) <= chunk_size:
                    new_sections.append(section)
                else:
                    sentences = re.split(r'(?<=[.!?])\s+', section)
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk + sentence) < chunk_size:
                            current_chunk += sentence + " "
                        else:
                            if current_chunk:
                                new_sections.append(current_chunk.strip())
                            current_chunk = sentence + " "
                    if current_chunk:
                        new_sections.append(current_chunk.strip())
            sections = new_sections
        
        return sections
    
    def get_final_knowledge_graph(self) -> Dict:
        """Get the final merged knowledge graph"""
        return {
            "edges": self.final_edges
        }
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the pipeline results"""
        base_subjects = set(edge.get("subject", "") for edge in self.base_edges)
        base_objects = set(edge.get("object", "") for edge in self.base_edges)
        base_entities = base_subjects.union(base_objects)
        
        aug_subjects = set(edge.get("subject", "") for edge in self.augmented_edges)
        aug_objects = set(edge.get("object", "") for edge in self.augmented_edges)
        aug_entities = aug_subjects.union(aug_objects)
        
        final_subjects = set(edge.get("subject", "") for edge in self.final_edges)
        final_objects = set(edge.get("object", "") for edge in self.final_edges)
        final_entities = final_subjects.union(final_objects)
        
        return {
            "step1_base_extraction": {
                "edges": len(self.base_edges),
                "entities": len(base_entities)
            },
            "step2_augmentation": {
                "new_edges": len(self.augmented_edges),
                "new_entities": len(aug_entities)
            },
            "step3_final_merged": {
                "total_edges": len(self.final_edges),
                "total_entities": len(final_entities),
                "augmentation_rate": f"{len(self.augmented_edges)/(len(self.base_edges)+1):.2%}"
            }
        }
    
    def save_knowledge_graph(self, filename: str = "knowledge_graph.json", save_intermediates: bool = True):
        """Save the knowledge graph and optionally intermediate results"""
        # Save final graph
        final_kg = self.get_final_knowledge_graph()
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(final_kg, f, indent=2, ensure_ascii=False)
        
        if save_intermediates:
            # Save intermediate results for analysis
            base_name = filename.replace('.json', '')
            
            # Save base extraction
            with open(f"{base_name}_step1_base.json", "w", encoding="utf-8") as f:
                json.dump({"edges": self.base_edges}, f, indent=2, ensure_ascii=False)
            
            # Save augmentation
            with open(f"{base_name}_step2_augmented.json", "w", encoding="utf-8") as f:
                json.dump({"edges": self.augmented_edges}, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Knowledge graph saved to {filename}")
        if save_intermediates:
            logger.info(f"Intermediate results saved with prefix: {base_name}")
        
        return filename

def main():
    """Main function demonstrating the complete pipeline"""
    
    # Initialize OpenAI client
    import os
    api_key = os.getenv("OPENAI_API_KEY") or "your-api-key-here"
    
    if api_key == "your-api-key-here":
        print("Please set your OpenAI API key!")
        print("Either set OPENAI_API_KEY environment variable or replace the key in the code")
        return
    
    client = OpenAI(api_key=api_key)
    
    # Sample text data
    with open("data/generated_sample.md", "r", encoding="utf-8") as f:
        sample_text = f.read()
    
    try:
        # Create knowledge graph builder using the complete pipeline
        kg_builder = KnowledgeGraphBuilder(client, model="gpt-4-turbo")
        
        logger.info("Starting complete graph construction pipeline...")
        
        # Run the complete pipeline
        kg_builder.process_text_with_pipeline(sample_text)
        
        # Get detailed statistics
        stats = kg_builder.get_pipeline_stats()
        
        print("\n=== PIPELINE RESULTS ===")
        print(f"Step 1 (Base Extraction): {stats['step1_base_extraction']['edges']} edges, {stats['step1_base_extraction']['entities']} entities")
        print(f"Step 2 (Augmentation): {stats['step2_augmentation']['new_edges']} new edges, {stats['step2_augmentation']['new_entities']} new entities")
        print(f"Step 3 (Final Merged): {stats['step3_final_merged']['total_edges']} total edges, {stats['step3_final_merged']['total_entities']} total entities")
        print(f"Augmentation Rate: {stats['step3_final_merged']['augmentation_rate']}")
        
        # Save results
        kg_builder.save_knowledge_graph("final_knowledge_graph.json", save_intermediates=True)
        
        # Display final result
        final_kg = kg_builder.get_final_knowledge_graph()
        print(f"\nFinal combined knowledge graph JSON:")
        # print(json.dumps(final_kg, indent=2, ensure_ascii=False))
        
    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        raise

if __name__ == "__main__":
    main()

    