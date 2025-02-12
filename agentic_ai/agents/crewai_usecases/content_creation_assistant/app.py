import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool, ScholarlyTool, SEOAnalyzerTool
from typing import List, Dict
import os
from datetime import datetime
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
LLM_CONFIGS = {
    "OpenAI": {
        "models": ["gpt-4o", "gpt-4o-mini"]
    },
    "Anthropic": {
        "models": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", 
                  "claude-3-opus-20240229"]
    },
    "Gemini": {
        "models": ["gemini-2.0-flash-exp", "gemini-1.5-flash", 
                  "gemini-1.5-flash-8b", "gemini-1.5-pro"]
    },
    "Groq": {
        "models": ["groq/deepseek-r1-distill-llama-70b", "groq/llama3-70b-8192", "groq/llama-3.1-8b-instant", 
                  "groq/llama-3.3-70b-versatile", "groq/gemma2-9b-it", "groq/mixtral-8x7b-32768"]
    }
}

class ContentCreationCrew:
    def __init__(self, llm_provider: str, model_name: str):
        """Initialize the Content Creation Crew with specific LLM configuration."""
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.llm = self._initialize_llm()
        self._setup_tools()
        
    def _initialize_llm(self) -> LLM:
        """Initialize the language model with appropriate configuration."""
        provider_keys = {
            "OpenAI": "OPENAI_API_KEY",
            "Anthropic": "ANTHROPIC_API_KEY",
            "Gemini": "GOOGLE_API_KEY",
            "Groq": "GROQ_API_KEY"
        }
        
        key_name = provider_keys.get(self.llm_provider)
        if key_name:
            os.environ[key_name] = os.getenv(key_name)
        
        return LLM(
            model=self.model_name,
            temperature=0.7,
            timeout=120,
            max_tokens=4000,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
        )

    def _setup_tools(self):
        """Initialize research and analysis tools."""
        self.search_tool = SerperDevTool()
        self.scholarly_tool = ScholarlyTool()
        self.seo_tool = SEOAnalyzerTool()

    def create_agents(self) -> List[Agent]:
        """Create specialized agents for content creation pipeline."""
        content_researcher = Agent(
            role='Content Researcher',
            goal='Conduct comprehensive research and gather authoritative sources',
            backstory="""Expert researcher with extensive experience in academic and 
            market research. Specialized in finding credible sources and identifying 
            key trends and insights.""",
            tools=[self.search_tool, self.scholarly_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=True
        )

        content_writer = Agent(
            role='Content Writer',
            goal='Create engaging, well-structured content optimized for target audience',
            backstory="""Professional writer with expertise in creating compelling 
            narratives and technical content. Strong understanding of various content 
            formats and audience engagement techniques.""",
            tools=[self.seo_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=True
        )

        content_editor = Agent(
            role='Content Editor',
            goal='Ensure content quality, accuracy, and SEO optimization',
            backstory="""Senior editor with strong attention to detail and expertise 
            in SEO best practices. Experienced in maintaining brand voice and editorial 
            standards.""",
            tools=[self.seo_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=True
        )

        return [content_researcher, content_writer, content_editor]

    def create_tasks(self, topic: str, content_type: str, target_audience: str,
                    content_requirements: Dict) -> List[Task]:
        """Create sequential tasks for the content creation pipeline."""
        
        researcher, writer, editor = self.create_agents()
        
        research_task = Task(
            description=f"""
            Conduct comprehensive research on: {topic}
            
            Content Type: {content_type}
            Target Audience: {target_audience}
            
            Research Requirements:
            1. Find authoritative sources and recent statistics
            2. Identify key trends and insights
            3. Analyze competitor content
            4. Gather relevant case studies or examples
            5. Document source credibility and relevance
            
            Additional Requirements:
            {content_requirements.get('research_requirements', '')}
            """,
            agent=researcher,
            expected_output="""Detailed research document including:
            - Key findings with citations
            - Market/industry analysis
            - Competitor content analysis
            - Relevant statistics and trends
            - Source credibility assessment"""
        )

        writing_task = Task(
            description=f"""
            Create {content_type} content based on research findings.
            
            Topic: {topic}
            Target Audience: {target_audience}
            Word Count: {content_requirements.get('word_count', '1000-1500')}
            
            Content Guidelines:
            1. Maintain clear structure and flow
            2. Include relevant examples and data
            3. Follow SEO best practices
            4. Incorporate target keywords
            5. Adapt tone for target audience
            
            Style Requirements:
            {content_requirements.get('style_requirements', '')}
            """,
            agent=writer,
            expected_output="""Well-structured content including:
            - Engaging introduction
            - Clear main points
            - Supporting evidence
            - Effective conclusion
            - Proper citations"""
        )

        editing_task = Task(
            description=f"""
            Review and optimize content for:
            1. Grammar and style consistency
            2. Technical accuracy
            3. SEO optimization
            4. Content flow and structure
            5. Citation accuracy
            
            Content Type: {content_type}
            Target Audience: {target_audience}
            
            Quality Requirements:
            {content_requirements.get('quality_requirements', '')}
            
            SEO Requirements:
            - Primary Keyword: {content_requirements.get('primary_keyword', '')}
            - Secondary Keywords: {content_requirements.get('secondary_keywords', [])}
            """,
            agent=editor,
            expected_output="""Polished content with:
            - Grammar and style corrections
            - SEO optimization report
            - Content quality assessment
            - Improvement recommendations
            - Final proofread version"""
        )

        return [research_task, writing_task, editing_task]

    def generate_content(self, topic: str, content_type: str, target_audience: str,
                        content_requirements: Dict) -> Dict:
        """Execute the content creation pipeline and return results."""
        try:
            # Create crew with tasks
            tasks = self.create_tasks(topic, content_type, target_audience, 
                                   content_requirements)
            
            crew = Crew(
                agents=self.create_agents(),
                tasks=tasks,
                verbose=True
            )

            # Execute the pipeline
            result = crew.kickoff()

            # Process and structure the results
            content_package = {
                'metadata': {
                    'topic': topic,
                    'content_type': content_type,
                    'target_audience': target_audience,
                    'timestamp': datetime.now().isoformat(),
                    'requirements': content_requirements
                },
                'content': str(result),
                'status': 'success'
            }

            return content_package

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'metadata': {
                    'topic': topic,
                    'timestamp': datetime.now().isoformat()
                }
            }

def render_workflow_diagram():
    """Render the Data Analysis Assistant workflow diagram."""
    with st.expander("📖 System Workflow", expanded=False):
        # Get the relative path to the image
        current_dir = Path(__file__).parent  # Directory of current script
        image_path = current_dir.parent.parent.parent.parent / 'images'
                
        routing_diagram = Image.open(image_path/ 'data_analysis_architecture.png')
        st.image(routing_diagram, caption='High Level Architecture')
                
        sequence_diagram = Image.open(image_path/ 'data_analysis_sequence_diagram.png')
        st.image(sequence_diagram, caption='Sequence Diagram')

def main():
    """Create Streamlit interface for the Content Creation Crew."""
    st.set_page_config(page_title="Content Creation Assistant", 
                      page_icon="📝", layout="wide")

    st.title("📝 Content Creation Assistant")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Content Settings")
        
        content_type = st.selectbox(
            "Content Type",
            ["Blog Post", "Technical Article", "White Paper", "Case Study", 
             "Newsletter", "Social Media Post"]
        )
        
        target_audience = st.selectbox(
            "Target Audience",
            ["General Reader", "Technical Professional", "Business Executive", 
             "Industry Expert", "Student"]
        )
        
        word_count = st.slider(
            "Target Word Count",
            min_value=500,
            max_value=5000,
            value=1000,
            step=500
        )

    # Main content area
    topic = st.text_area(
        "Content Topic",
        placeholder="Enter your content topic...",
        height=100
    )

    col1, col2 = st.columns(2)
    with col1:
        primary_keyword = st.text_input("Primary Keyword")
        style_requirements = st.text_area(
            "Style Requirements",
            placeholder="Enter any specific style guidelines..."
        )

    with col2:
        secondary_keywords = st.text_area(
            "Secondary Keywords (one per line)",
            placeholder="Enter secondary keywords..."
        )
        quality_requirements = st.text_area(
            "Quality Requirements",
            placeholder="Enter quality control specifications..."
        )

    if st.button("Generate Content", type="primary"):
        if not topic:
            st.error("Please enter a content topic")
            return

        content_requirements = {
            'word_count': word_count,
            'primary_keyword': primary_keyword,
            'secondary_keywords': secondary_keywords.split('\n') if secondary_keywords else [],
            'style_requirements': style_requirements,
            'quality_requirements': quality_requirements
        }

        crew = ContentCreationCrew("Anthropic", "claude-3-opus-20240229")
        
        with st.spinner("Generating content..."):
            result = crew.generate_content(
                topic=topic,
                content_type=content_type,
                target_audience=target_audience,
                content_requirements=content_requirements
            )

        if result['status'] == 'success':
            st.success("Content generated successfully!")
            
            # Display tabs for different views
            tab1, tab2 = st.tabs(["📄 Content", "📊 Metadata"])
            
            with tab1:
                st.markdown(result['content'])
                
                # Download button
                st.download_button(
                    label="Download Content",
                    data=result['content'],
                    file_name=f"content_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
            
            with tab2:
                st.json(result['metadata'])
        else:
            st.error(f"Error generating content: {result['error']}")

if __name__ == "__main__":
    main()