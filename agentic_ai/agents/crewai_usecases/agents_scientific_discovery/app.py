import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain.tools import Tool
from typing import Dict, List
import json
import os
import random
import time
from datetime import datetime
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

# Configuration
LLM_CONFIGS = {
    "OpenAI": {
        "models": ["gpt-4", "gpt-3.5-turbo"],
        "api_key": "OPENAI_API_KEY"
    }
}

# OWASP Security Guidelines
OWASP_GUIDELINES = {
    "DDoS Attack": {
        "description": "Distributed Denial of Service attack attempting to overwhelm systems",
        "mitigations": [
            "Implement rate limiting",
            "Use DDoS protection services",
            "Configure network threshold alerts",
            "Deploy traffic filtering"
        ],
        "risk_level": "Critical"
    },
    "Phishing Attempt": {
        "description": "Attempt to steal sensitive information through deception",
        "mitigations": [
            "Enable email filtering",
            "Implement SPF/DKIM/DMARC",
            "User security awareness training",
            "Deploy anti-phishing solutions"
        ],
        "risk_level": "High"
    },
    "Malware Injection": {
        "description": "Attempt to insert malicious code into the system",
        "mitigations": [
            "Keep systems patched and updated",
            "Use robust antivirus solutions",
            "Implement application whitelisting",
            "Regular security scans"
        ],
        "risk_level": "Critical"
    },
    "SQL Injection": {
        "description": "Attempt to inject malicious SQL code",
        "mitigations": [
            "Use parameterized queries",
            "Input validation and sanitization",
            "Implement WAF",
            "Regular security testing"
        ],
        "risk_level": "Critical"
    },
    "Brute Force Login": {
        "description": "Systematic attempt to guess login credentials",
        "mitigations": [
            "Implement account lockout",
            "Use strong password policies",
            "Enable 2FA/MFA",
            "Monitor login attempts"
        ],
        "risk_level": "High"
    }
}

def generate_dummy_network_traffic():
    """Simulates network traffic logs with occasional threats."""
    threats = list(OWASP_GUIDELINES.keys())
    traffic_logs = []
    for i in range(10):
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_ip": f"192.168.1.{random.randint(1, 255)}",
            "destination_ip": "10.0.0.1",
            "port": random.randint(1, 65535),
            "protocol": random.choice(["TCP", "UDP", "HTTP", "HTTPS"]),
            "bytes_transferred": random.randint(100, 10000),
            "status": "Normal Traffic"
        }
        if random.random() < 0.3:  # 30% chance of threat
            threat_type = random.choice(threats)
            log_entry["status"] = threat_type
            log_entry["risk_level"] = OWASP_GUIDELINES[threat_type]["risk_level"]
        traffic_logs.append(log_entry)
        time.sleep(0.1)
    return traffic_logs

class NetworkData:
    """Class to handle network traffic generation and basic processing"""
    @staticmethod
    def generate_dummy_traffic():
        """Generates simulated network traffic"""
        return generate_dummy_network_traffic()

    @staticmethod
    def format_traffic_for_analysis(logs):
        """Formats traffic data for agent consumption"""
        try:
            # Ensure logs is a list
            if not isinstance(logs, list):
                if isinstance(logs, str):
                    logs = json.loads(logs)
                if not isinstance(logs, list):
                    logs = [logs]
            
            formatted_data = {
                "traffic_data": logs
            }
            return json.dumps(formatted_data)
        except Exception as e:
            return json.dumps({
                "error": f"Error formatting traffic data: {str(e)}",
                "traffic_data": []
            })

    @staticmethod
    def analyze_traffic_patterns(input_data: str) -> str:
        """Analyzes traffic patterns and returns structured data as JSON string"""
        try:
            # Parse input data
            if isinstance(input_data, str):
                data = json.loads(input_data)
            else:
                data = input_data

            # Extract traffic data
            if isinstance(data, dict):
                traffic_data = data.get("traffic_data", [])
                if isinstance(traffic_data, str):
                    traffic_data = json.loads(traffic_data)
            elif isinstance(data, list):
                traffic_data = data
            else:
                traffic_data = [data]

            # Ensure traffic_data is a list
            if not isinstance(traffic_data, list):
                traffic_data = [traffic_data]

            # Initialize patterns
            patterns = {
                "protocols": {},
                "threat_distribution": {},
                "peak_traffic": 0,
                "suspicious_ips": []
            }
            
            # Analyze each log entry
            for entry in traffic_data:
                if not isinstance(entry, dict):
                    continue
                
                # Protocol analysis
                protocol = entry.get("protocol", "unknown")
                patterns["protocols"][protocol] = patterns["protocols"].get(protocol, 0) + 1
                
                # Traffic volume
                bytes_transferred = entry.get("bytes_transferred", 0)
                if isinstance(bytes_transferred, (int, float)):
                    patterns["peak_traffic"] = max(patterns["peak_traffic"], bytes_transferred)
                
                # Threat tracking
                status = entry.get("status", "Normal Traffic")
                if status != "Normal Traffic":
                    patterns["threat_distribution"][status] = \
                        patterns["threat_distribution"].get(status, 0) + 1
                    source_ip = entry.get("source_ip")
                    if source_ip and source_ip not in patterns["suspicious_ips"]:
                        patterns["suspicious_ips"].append(source_ip)
            
            return json.dumps(patterns)

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return json.dumps({
                "error": f"Error analyzing traffic: {str(e)}\nDetail: {error_detail}",
                "patterns": {
                    "protocols": {},
                    "threat_distribution": {},
                    "peak_traffic": 0,
                    "suspicious_ips": []
                }
            })

class SecurityTools:
    """Collection of tools for security analysis"""
    def __init__(self, serper_api_key):
        self.search_tool = SerperDevTool(api_key=serper_api_key)
        
        def get_owasp_guidelines(threat_type: str) -> str:
            """Get OWASP guidelines for a specific threat type"""
            try:
                if isinstance(threat_type, dict):
                    # Extract threat type from dict if needed
                    threat_type = threat_type.get('threat_type', '')
                # Normalize threat type string
                threat_type = str(threat_type).strip()
                guidelines = OWASP_GUIDELINES.get(threat_type, {})
                return json.dumps({
                    'threat_type': threat_type,
                    'guidelines': guidelines
                })
            except Exception as e:
                return json.dumps({
                    'error': str(e),
                    'message': f'Failed to get OWASP guidelines for {threat_type}'
                })
        
        self.owasp_tool = Tool(
            name="OWASP_Guidelines",
            func=get_owasp_guidelines,
            description="Look up OWASP guidelines for specific threat types"
        )
        
        def analyze_traffic(traffic_data: str) -> str:
            """Wrapper function to ensure proper data handling"""
            try:
                # Handle both string and dict inputs
                if isinstance(traffic_data, dict):
                    traffic_data = json.dumps(traffic_data)
                return NetworkData.analyze_traffic_patterns(traffic_data)
            except Exception as e:
                return json.dumps({
                    'error': str(e),
                    'message': 'Failed to analyze traffic data'
                })
        
        self.traffic_analyzer = Tool(
            name="Traffic_Analysis",
            func=analyze_traffic,
            description="Analyze network traffic patterns for anomalies"
        )

class SecurityAnalyzer:
    def __init__(self, llm_provider: str, model_name: str):
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.tools = SecurityTools(os.environ.get("SERPER_API_KEY"))
        
    def create_agents(self) -> List[Agent]:
        """Creates specialized security agents with specific roles and tools"""
        
        network_monitor = Agent(
            role='Network Monitor',
            goal='Monitor network traffic and identify potential security threats',
            backstory="""Expert in network security monitoring with extensive experience in 
            threat detection. You analyze traffic patterns and identify anomalies that could 
            indicate security threats.""",
            tools=[self.tools.traffic_analyzer],
            verbose=True
        )
        
        threat_analyzer = Agent(
            role='Threat Analyzer',
            goal='Analyze detected threats and assess their potential impact',
            backstory="""Senior security analyst specializing in threat assessment. You take 
            identified anomalies and determine if they represent genuine threats, assessing 
            their potential impact and severity.""",
            tools=[self.tools.search_tool, self.tools.owasp_tool],
            verbose=True
        )
        
        response_coordinator = Agent(
            role='Response Coordinator',
            goal='Develop and coordinate security responses based on threat analysis',
            backstory="""Security operations expert focused on incident response. You take 
            analyzed threats and develop specific response strategies, coordinating with 
            other team members to ensure effective threat mitigation.""",
            tools=[self.tools.owasp_tool],
            verbose=True
        )
        
        security_advisor = Agent(
            role='OWASP Security Advisor',
            goal='Provide expert security recommendations based on OWASP guidelines',
            backstory="""OWASP certified security consultant with deep expertise in web 
            application security. You provide specific guidance based on OWASP standards 
            and best practices.""",
            tools=[self.tools.owasp_tool, self.tools.search_tool],
            verbose=True
        )
        
        return [network_monitor, threat_analyzer, response_coordinator, security_advisor]

    def create_tasks(self, agents, traffic_data) -> List[Task]:
        """Creates sequential tasks for security analysis"""
        
        # Format traffic data for analysis
        base_traffic_data = {
            "traffic_data": traffic_data
        }

        # Network Monitor Task
        monitor_task = Task(
            description=f"""Analyze this network traffic data for potential security threats. 
            Traffic data summary: {len(traffic_data)} entries.
            
            Your task:
            1. Review all traffic patterns
            2. Flag any suspicious activities
            3. Identify potential threat indicators
            4. Categorize findings by severity
            
            When using the Traffic_Analysis tool:
            - The input should be a JSON string containing traffic data
            - Each traffic entry should have protocol, status, and bytes_transferred fields
            
            When using the OWASP_Guidelines tool:
            - Input should be a simple string with the threat type
            - Example: "DDoS Attack" or "SQL Injection"
            
            Provide detailed explanation for each flagged activity.""",
            expected_output="""A structured JSON report containing:
            - Identified threats
            - Suspicious patterns
            - Traffic anomalies
            - Severity classifications""",
            agent=agents[0],  # Network Monitor
            context=[
                {
                    "role": "user",
                    "content": "Analyze this network traffic for security threats.",
                    "description": "Network traffic analysis task",
                    "expected_output": "Security threat analysis report",
                    "input_data": json.dumps(base_traffic_data)
                }
            ]
        )
        
        # Threat Analyzer Task
        analyze_task = Task(
            description="""Based on the monitoring results, perform deep analysis of identified threats.
            
            Your task:
            1. Research each identified threat using SerperDev
            2. Compare against known attack patterns
            3. Assess potential impact
            4. Determine threat severity
            5. Document evidence and indicators
            
            Provide comprehensive analysis with supporting data.""",
            expected_output="""A detailed threat analysis report containing:
            - Threat profiles
            - Impact assessments
            - Severity ratings
            - Supporting evidence""",
            agent=agents[1],  # Threat Analyzer
            context=[
                {
                    "role": "user",
                    "content": "Analyze these identified threats in detail.",
                    "description": "Detailed threat analysis task",
                    "expected_output": "Comprehensive threat assessment",
                    "previous_results": "Will be populated from monitor_task"
                }
            ]
        )
        
        # Response Coordinator Task
        response_task = Task(
            description="""Develop detailed response strategies for confirmed threats.
            
            Your task:
            1. Review threat analysis
            2. Consult OWASP guidelines
            3. Develop specific mitigation steps
            4. Prioritize response actions
            5. Create timeline for implementation
            
            Provide actionable response plan with clear steps.""",
            expected_output="""A structured response plan containing:
            - Mitigation strategies
            - Prioritized actions
            - Implementation timeline
            - Resource requirements""",
            agent=agents[2],  # Response Coordinator
            context=[
                {
                    "role": "user",
                    "content": "Develop response strategies for these threats.",
                    "description": "Response strategy development task",
                    "expected_output": "Detailed response plan",
                    "threat_analysis": "Will be populated from analyze_task"
                }
            ]
        )
        
        # Security Advisor Task
        advisory_task = Task(
            description="""Review all findings and provide expert OWASP-based recommendations.
            
            Your task:
            1. Review all previous analyses
            2. Compare against OWASP standards
            3. Identify gaps in proposed responses
            4. Suggest additional security measures
            5. Provide long-term security recommendations
            
            Deliver comprehensive security advisory report.""",
            expected_output="""A comprehensive security advisory containing:
            - OWASP compliance assessment
            - Gap analysis
            - Additional security recommendations
            - Long-term security roadmap""",
            agent=agents[3],  # Security Advisor
            context=[
                {
                    "role": "user",
                    "content": "Review all findings and provide OWASP-based recommendations.",
                    "description": "Security advisory task",
                    "expected_output": "Comprehensive security recommendations",
                    "previous_analyses": {
                        "monitoring": "Will be populated from monitor_task",
                        "analysis": "Will be populated from analyze_task",
                        "response": "Will be populated from response_task"
                    }
                }
            ]
        )
        
        return [monitor_task, analyze_task, response_task, advisory_task]

    def run_security_analysis(self):
        """Runs the complete security analysis process"""
        
        # Generate traffic data
        traffic_data = NetworkData.generate_dummy_traffic()
        
        # Create agents and tasks
        agents = self.create_agents()
        tasks = self.create_tasks(agents, traffic_data)
        
        # Create crew with sequential process
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
        
        # Execute crew tasks
        results = crew.kickoff()
        return results, traffic_data

def create_network_visualizations(logs: List[Dict], analysis_results: Dict) -> Dict[str, go.Figure]:
    """Create visualizations for network traffic and analysis results"""
    visualizations = {}
    
    # Convert logs to DataFrame
    df_logs = pd.DataFrame(logs)
    
    # Traffic Timeline
    fig_timeline = px.scatter(
        df_logs,
        x='timestamp',
        y='bytes_transferred',
        color='status',
        title='Network Traffic Timeline',
        hover_data=['source_ip', 'protocol', 'port'],
        color_discrete_map={
            'Normal Traffic': 'green',
            'DDoS Attack': 'red',
            'Phishing Attempt': 'orange',
            'Malware Injection': 'purple',
            'SQL Injection': 'brown',
            'Brute Force Login': 'pink'
        }
    )
    visualizations['traffic_timeline'] = fig_timeline
    
    # Protocol Distribution
    protocol_dist = df_logs.groupby('protocol').size()
    fig_protocol = px.pie(
        values=protocol_dist.values,
        names=protocol_dist.index,
        title='Protocol Distribution'
    )
    visualizations['protocol_dist'] = fig_protocol
    
    # Threat Distribution
    threats = df_logs[df_logs['status'] != 'Normal Traffic']
    if not threats.empty:
        threat_dist = threats.groupby('status').size()
        fig_threats = px.bar(
            x=threat_dist.index,
            y=threat_dist.values,
            title='Threat Distribution',
            labels={'x': 'Threat Type', 'y': 'Count'}
        )
        visualizations['threat_dist'] = fig_threats
    
    return visualizations

def display_agent_findings(findings: Dict, container):
    """Displays agent findings in the Streamlit UI"""
    for agent, result in findings.items():
        with container.expander(f"🔍 {agent} Findings", expanded=True):
            st.markdown("**Key Findings:**")
            if isinstance(result, dict):
                for key, value in result.items():
                    st.markdown(f"**{key}:**")
                    st.write(value)
            else:
                st.write(result)

def convert_crew_output_to_dict(crew_output):
    """Convert CrewOutput object to a serializable dictionary"""
    try:
        if hasattr(crew_output, 'dict'):
            # If object has a dict method, use it
            return crew_output.dict()
        elif hasattr(crew_output, '__dict__'):
            # If object has __dict__, convert it
            return {
                key: convert_crew_output_to_dict(value) 
                if hasattr(value, '__dict__') else value
                for key, value in crew_output.__dict__.items()
            }
        elif isinstance(crew_output, list):
            # If it's a list, convert each item
            return [convert_crew_output_to_dict(item) for item in crew_output]
        elif isinstance(crew_output, dict):
            # If it's a dict, convert each value
            return {
                key: convert_crew_output_to_dict(value) 
                if hasattr(value, '__dict__') else value
                for key, value in crew_output.items()
            }
        else:
            # If it's a primitive type, return as is
            return crew_output
    except Exception as e:
        return str(crew_output)  # Fallback to string representation

def parse_crew_output(output):
    """Parse CrewAI output into a structured format"""
    if isinstance(output, str):
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {"raw_output": output}
    
    # Handle case where output is already a dictionary
    if isinstance(output, dict):
        return output
        
    parsed_results = []
    current_agent = None
    current_result = {}
    
    # Process each line of the output
    for line in str(output).split('\n'):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Check for agent headers
        if '# Agent:' in line:
            if current_agent and current_result:
                parsed_results.append({
                    'agent': current_agent,
                    **current_result
                })
            current_agent = line.replace('# Agent:', '').strip()
            current_result = {}
            continue
            
        # Check for final answer
        if '## Final Answer:' in line:
            try:
                # Try to parse as JSON if it starts with {
                content = line.replace('## Final Answer:', '').strip()
                if content.startswith('{'):
                    current_result['final_answer'] = json.loads(content)
                else:
                    current_result['final_answer'] = content
            except json.JSONDecodeError:
                current_result['final_answer'] = line.replace('## Final Answer:', '').strip()
            continue
            
        # Check for thought process
        if '## Thought:' in line:
            if 'thoughts' not in current_result:
                current_result['thoughts'] = []
            current_result['thoughts'].append(line.replace('## Thought:', '').strip())
            continue
            
        # Check for tool usage
        if '## Tool Input:' in line:
            if 'tools' not in current_result:
                current_result['tools'] = []
            current_result['tools'].append({
                'input': line.replace('## Tool Input:', '').strip()
            })
            continue
            
        if '## Tool Output:' in line:
            if current_result.get('tools'):
                current_result['tools'][-1]['output'] = line.replace('## Tool Output:', '').strip()
            continue
    
    # Add the last agent's results
    if current_agent and current_result:
        parsed_results.append({
            'agent': current_agent,
            **current_result
        })
    
    return parsed_results

def display_agent_results(results, container):
    """Display agent results in the Streamlit container"""
    parsed_results = parse_crew_output(results)
    print(f"##############################{results}")
    if not parsed_results:
        container.warning("No results available to display.")
        return
        
    for result in parsed_results:
        agent_name = result.get('agent', 'Unknown Agent')
        with container.expander(f"🔍 {agent_name}", expanded=True):
            # Display thoughts
            if result.get('thoughts'):
                container.markdown("### Thought Process")
                for thought in result['thoughts']:
                    container.markdown(f"- {thought}")
            
            # Display tool usage
            if result.get('tools'):
                container.markdown("### Tools Used")
                for tool in result['tools']:
                    container.markdown("**Input:**")
                    container.code(tool['input'], language='json')
                    if 'output' in tool:
                        container.markdown("**Output:**")
                        container.code(tool['output'], language='json')
            
            # Display final answer
            if result.get('final_answer'):
                container.markdown("### Final Analysis")
                if isinstance(result['final_answer'], dict):
                    container.json(result['final_answer'])
                elif isinstance(result['final_answer'], str):
                    # Check if the string is JSON
                    try:
                        container.json(json.loads(result['final_answer']))
                    except json.JSONDecodeError:
                        container.markdown(result['final_answer'])

def display_response_plan(results, container):
    """Display response plan in the Streamlit container"""
    parsed_results = parse_crew_output(results)
    
    # Filter for Response Coordinator results
    response_results = [r for r in parsed_results if 'Response Coordinator' in r.get('agent', '')]
    
    if not response_results:
        container.warning("No response plan available.")
        return
        
    for result in response_results:
        with container.expander("🛡️ Response Plan", expanded=True):
            if result.get('final_answer'):
                container.markdown("### Recommended Actions")
                if isinstance(result['final_answer'], dict):
                    container.json(result['final_answer'])
                else:
                    container.markdown(result['final_answer'])
            
            if result.get('thoughts'):
                container.markdown("### Analysis Process")
                for thought in result['thoughts']:
                    container.markdown(f"- {thought}")

def display_agent_logs(results, container):
    """Display detailed agent logs in the Streamlit container"""
    parsed_results = parse_crew_output(results)
    
    if not parsed_results:
        container.warning("No agent logs available.")
        return
        
    for result in parsed_results:
        agent_name = result.get('agent', 'Unknown Agent')
        with container.expander(f"📋 {agent_name} Logs", expanded=True):
            # Display thoughts
            if result.get('thoughts'):
                container.markdown("### Thought Process")
                for thought in result['thoughts']:
                    container.markdown(f"- {thought}")
            
            # Display tool usage with proper formatting
            if result.get('tools'):
                container.markdown("### Tools Used")
                for idx, tool in enumerate(result['tools'], 1):
                    container.markdown(f"**Tool Usage #{idx}**")
                    container.markdown("*Input:*")
                    container.code(tool['input'], language='json')
                    if 'output' in tool:
                        container.markdown("*Output:*")
                        container.code(tool['output'], language='json')
                    container.markdown("---")
            
            # Display final answer if available
            if result.get('final_answer'):
                container.markdown("### Final Output")
                if isinstance(result['final_answer'], dict):
                    container.json(result['final_answer'])
                else:
                    container.markdown(result['final_answer'])

def main():
    st.set_page_config(
        page_title="Enhanced Cybersecurity Monitoring",
        page_icon="🔒",
        layout="wide"
    )
    
    st.header("🔒 Cybersecurity Monitoring")
    st.markdown("Agentic Security Analysis using CrewAI Multi-Agents")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        llm_provider = st.selectbox(
            "Select LLM Provider",
            options=list(LLM_CONFIGS.keys())
        )
        
        selected_model = st.selectbox(
            "Select Model",
            options=LLM_CONFIGS[llm_provider]["models"]
        )
        
        st.markdown("---")
        st.markdown("### OWASP Guidelines")
        if st.checkbox("Show OWASP Guidelines"):
            for threat, details in OWASP_GUIDELINES.items():
                with st.expander(f"{threat} ({details['risk_level']})"):
                    st.write(details['description'])
                    st.write("**Mitigations:**")
                    for mitigation in details['mitigations']:
                        st.write(f"• {mitigation}")
    
    # Initialize analyzer
    analyzer = SecurityAnalyzer(llm_provider, selected_model)
    
    if st.button("Start Security Analysis"):
        with st.spinner("Running agentic security analysis..."):
            # Run analysis
            results, traffic_data = analyzer.run_security_analysis()
            
            # Create tabs for different views
            tabs = st.tabs([
                "📊 Live Monitor",
                "🔍 Analysis Results",
                "🛡️ Response Plan",
                "📋 Agent Logs"
            ])
            
            # Live Monitor tab
            with tabs[0]:
                st.subheader("Network Traffic Monitor")
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Traffic", len(traffic_data))
                with col2:
                    threats = len([log for log in traffic_data if log["status"] != "Normal Traffic"])
                    st.metric("Active Threats", threats)
                with col3:
                    critical = len([log for log in traffic_data if log.get("risk_level") == "Critical"])
                    st.metric("Critical Threats", critical)
                with col4:
                    protocols = len(set(log["protocol"] for log in traffic_data))
                    st.metric("Unique Protocols", protocols)
                
                # Display traffic data
                st.markdown("### Live Network Traffic")
                traffic_df = pd.DataFrame(traffic_data)
                st.dataframe(
                    traffic_df,
                    use_container_width=True,
                    height=200
                )
                
                # Display visualizations
                visualizations = create_network_visualizations(traffic_data, results)
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(visualizations['traffic_timeline'], use_container_width=True)
                with col2:
                    st.plotly_chart(visualizations['protocol_dist'], use_container_width=True)
                if 'threat_dist' in visualizations:
                    st.plotly_chart(visualizations['threat_dist'], use_container_width=True)
            
            # Update this part in your main() function to properly handle agent results:

             # Analysis Results tab
            with tabs[1]:
                st.subheader("Agent Analysis Results")
                display_agent_results(results, st)
            
            # Response Plan tab
            with tabs[2]:
                st.subheader("Security Response Plan")
                display_response_plan(results, st)
            
            # Agent Logs tab
            with tabs[3]:
                st.subheader("AI Agent Activity Logs")
                display_agent_logs(results, st)
                
            # Generate downloadable report
            st.markdown("---")
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'analysis_summary': {
                    'total_traffic': len(traffic_data),
                    'total_threats': len([log for log in traffic_data if log["status"] != "Normal Traffic"]),
                    'critical_threats': len([log for log in traffic_data if log.get("risk_level") == "Critical"]),
                    'unique_protocols': len(set(log["protocol"] for log in traffic_data))
                },
                'agent_findings': convert_crew_output_to_dict(results),
                'traffic_data': traffic_data
            }
            
            # Add download button
            st.download_button(
                label="Download Analysis Report",
                data=json.dumps(report_data, indent=2),
                file_name=f"security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()