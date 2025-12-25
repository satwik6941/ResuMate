import requests
import json
import time
import re
from typing import Dict, List, Optional, Any
try:
    from googlesearch import search
except ImportError:
    print("Warning: googlesearch-python not installed. Search functionality will be limited.")
    search = None

class GroqLLM:
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def _make_request(self, messages: List[Dict], max_tokens: int = 2000, 
                        temperature: float = 0.7, timeout: int = 30) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                elif response.status_code == 429: 
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (2 ** attempt))
                        continue
                    else:
                        return "❌ Rate limit exceeded. Please try again later."
                else:
                    return f"❌ API Error {response.status_code}: {response.text}"
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    return "❌ Request timeout. Please try again."
            except requests.exceptions.RequestException as e:
                return f"❌ Network error: {str(e)}"
            except Exception as e:
                return f"❌ Unexpected error: {str(e)}"
        
        return "❌ Failed to get response after multiple attempts."
    
    def search_unknown_terms(self, text: str, context: str = "") -> Dict[str, str]:
        if not search:
            return {}
            
        unknown_terms = self._extract_unknown_terms(text)
        search_results = {}
        
        for term in unknown_terms[:5]: 
            try:           
                search_query = f"{term} definition meaning {context}" if context else f"{term} definition meaning"
                results = list(search(search_query, num_results=2, sleep_interval=2))
                
                if results:
                    explanation = self._get_term_explanation(term, search_query)
                    if explanation and len(explanation) > 10:
                        search_results[term] = explanation
                        
            except Exception as e:
                print(f"Search error for term '{term}': {e}")
                continue
                
        return search_results
    
    def _extract_unknown_terms(self, text: str) -> List[str]:
        patterns = [
            r'\b[A-Z]{2,}\b',  
            r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', 
            r'\b\w+(?:\.js|\.py|\.net|\.io|\.com)\b',  
            r'\b(?:React|Angular|Vue|Django|Flask|Laravel|Spring|Docker|Kubernetes|TensorFlow|PyTorch)\b',  # Common tech terms
        ]
        
        terms = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.update(matches)
            
        common_words = {'The', 'This', 'That', 'And', 'Or', 'But', 'For', 'In', 'On', 'At', 'To', 'Of', 'API', 'UI', 'UX'}
        filtered_terms = []
        for term in terms:
            if (term not in common_words and 
                len(term) > 2 and 
                not term.isdigit() and 
                term.upper() not in ['CEO', 'CTO', 'HR', 'IT']):
                filtered_terms.append(term)
                
        return filtered_terms[:10]
    
    def _get_term_explanation(self, term: str, search_query: str) -> str:
        try:
            prompt = f"""
            Provide a brief, clear explanation of the term "{term}" in 1-2 sentences.
            Focus on what it means in a professional/technical context.
            Be concise and accurate.
            """
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant that provides clear, concise explanations of technical and professional terms."},
                {"role": "user", "content": prompt}
            ]
            
            explanation = self._make_request(messages, max_tokens=100, temperature=0.3)
            return explanation if explanation and not explanation.startswith("❌") else ""
        except Exception:
            return ""

    def generate_enhanced_cover_letter(self, user_data: Dict[str, Any], job_description: str, 
                                        company_name: str, tone: str = "Professional") -> str:
        
        user_skills = user_data.get('skills', [])
        user_experience = user_data.get('experience', [])
        user_projects = user_data.get('projects', [])
        
        experience_context = ""
        if user_experience:
            experience_context = "\n📋 PROFESSIONAL EXPERIENCE:\n"
            for exp in user_experience[:3]: 
                if isinstance(exp, dict):
                    position = exp.get('position') or '[Position details to be specified]'
                    company = exp.get('company') or '[Company name to be specified]'
                    duration = exp.get('duration') or '[Duration to be specified]'
                    description = exp.get('description') or '[Experience details to be highlighted]'
                    
                    experience_context += f"• {position} at {company}\n"
                    experience_context += f"  Duration: {duration}\n"
                    experience_context += f"  Achievements: {description[:200]}...\n\n"
        
        projects_context = ""
        if user_projects:
            projects_context = "\n🚀 KEY PROJECTS:\n"
            for project in user_projects[:3]:
                if isinstance(project, dict):
                    title = project.get('title') or '[Project title to be specified]'
                    description = project.get('description') or '[Project description to be provided]'
                    technologies = project.get('technologies') or ''
                    
                    projects_context += f"• {title}\n"
                    projects_context += f"  Description: {description[:150]}...\n"
                    if technologies:
                        projects_context += f"  Technologies: {technologies}\n"
                    projects_context += "\n"
        
        skills_context = ""
        if user_skills:
            skills_context = f"\n🛠️ TECHNICAL SKILLS: {', '.join(user_skills[:15])}"
        
        search_results = self.search_unknown_terms(job_description, f"{company_name} job requirements technical skills")
        
        search_context = ""
        if search_results:
            search_context = "\n\n🔬 INDUSTRY RESEARCH:\n"
            for term, explanation in search_results.items():
                search_context += f"• {term}: {explanation}\n"
        
        job_analysis_prompt = f"""
        Analyze this job description and identify:
        1. Key technical requirements
        2. Required experience level  
        3. Company culture indicators
        4. Essential soft skills
        5. Industry-specific terminology
        
        Job Description: {job_description[:1000]}
        
        Return a brief analysis of what this role needs most.
        """
        
        job_analysis = self._make_request([
            {"role": "system", "content": "You are a job requirements analyst. Provide concise analysis."},
            {"role": "user", "content": job_analysis_prompt}
        ], max_tokens=300, temperature=0.3)
        
        prompt = f"""
        Write a compelling, {tone.lower()} cover letter that demonstrates clear alignment between the candidate's background and the job requirements.
        
        🎯 TARGET ROLE: {company_name} - Position from job description
        👤 CANDIDATE PROFILE:
        Name: {user_data.get('name', 'Candidate Name Not Provided')}
        Current Title: {user_data.get('title') or 'Title Not Specified'}
        Location: {user_data.get('location') or 'Location Flexible'}
        Summary: {user_data.get('summary') or 'Professional background details to be highlighted'}
        {skills_context}
        {experience_context}
        {projects_context}
        
        📋 JOB REQUIREMENTS ANALYSIS:
        {job_analysis}
        
        📄 FULL JOB DESCRIPTION:
        {job_description}
        
        {search_context}
        
        🎨 WRITING GUIDELINES:
        Create a personalized cover letter that:
        
        1. **Opening Hook**: Start with a compelling connection to the company or role that shows research
        2. **Skills Alignment**: Map specific user skills/experience to job requirements with concrete examples
        3. **Project Showcase**: Highlight relevant projects that demonstrate required capabilities
        4. **Company Knowledge**: Show understanding of company culture, recent news, or industry position
        5. **Value Proposition**: Clearly articulate what unique value the candidate brings
        6. **Technical Depth**: Include relevant technical terminology and concepts from research
        7. **Achievement Focus**: Quantify accomplishments where possible from user's background
        8. **Cultural Fit**: Demonstrate alignment with company values and work style
        9. **Call to Action**: End with confidence and next steps
        
        📐 FORMATTING REQUIREMENTS:
        - 350-450 words total
        - {tone} tone throughout
        - Natural integration of researched terminology
        - Specific examples from user's actual background
        - No generic statements or placeholders
        - Professional but engaging language
        
        🚫 AVOID:
        - Generic phrases like "I am writing to apply"
        - Repetition of resume content without added insight
        - Overly formal or robotic language
        - Claims not supported by user's background
        - Template-like structure
        
        Write as if you're the candidate, using their real experience and skills to create an authentic, compelling case for why they're the perfect fit for this specific role at this specific company.
        """
        
        messages = [
            {"role": "system", "content": f"You are an expert career counselor who writes compelling, {tone.lower()} cover letters that get interviews. You have access to current industry knowledge and create highly personalized content."},
            {"role": "user", "content": prompt}
        ]
        
        return self._make_request(messages, max_tokens=1500, temperature=0.7)

    def generate_enhanced_resume(self, user_data: Dict[str, Any]) -> str:
        style = user_data.get('resume_style', 'Professional ATS-Optimized')
        skills_input = user_data.get('skills_input', '')
        projects = user_data.get('projects', [])
        
        all_content = f"""
        {user_data.get('summary', '')} 
        {skills_input} 
        {user_data.get('experience', '')} 
        {user_data.get('title', '')}
        {' '.join([str(exp.get('description', '')) for exp in user_data.get('experience', []) if isinstance(exp, dict)])}
        {' '.join([str(proj.get('description', '')) for proj in projects if isinstance(proj, dict)])}
        """
        
        search_results = self.search_unknown_terms(all_content, "career skills technology industry trends")
        
        search_context = ""
        if search_results:
            search_context = "\n\n🔬 ENHANCED TECHNICAL KNOWLEDGE:\n"
            for term, explanation in search_results.items():
                search_context += f"• {term}: {explanation}\n"
        
        projects_context = ""
        if projects:
            projects_context = "\n\n📚 PROJECTS TO SHOWCASE (Transform using STAR methodology):\n"
            for i, project in enumerate(projects, 1):
                projects_context += f"\nProject {i}:\n"
                projects_context += f"• Title: {project.get('title', 'Untitled Project')}\n"
                projects_context += f"• Description: {project.get('description', 'No description provided')}\n"
                if project.get('technologies'):
                    projects_context += f"• Technologies: {project.get('technologies')}\n"
                if project.get('duration'):
                    projects_context += f"• Duration: {project.get('duration')}\n"
        
        prompt = f"""
        Create an enhanced, {style} resume for:
        
        Name: {user_data.get('name') or '[Name to be provided]'}
        Title: {user_data.get('title') or '[Professional title to be specified]'}
        Email: {user_data.get('email') or '[Email address to be provided]'}
        Phone: {user_data.get('phone') or '[Phone number to be provided]'}
        Skills: {', '.join(user_data.get('skills', [])) if user_data.get('skills') else '[Skills to be specified]'}
        Experience: {user_data.get('experience') or '[Professional experience to be detailed]'}
        Education: {user_data.get('education') or '[Educational background to be provided]'}
        
        {search_context}
        {projects_context}
        Create a professional resume with:
        1. Compelling professional summary using current industry terminology
        2. Quantified achievements with metrics
        3. Industry-specific keywords (informed by research above)
        4. ATS-optimized formatting
        5. Strong action verbs
        6. Relevant technical skills highlighted with proper context
        7. Current industry trends and terminology integration
        8. Projects section formatted using STAR method (Situation, Task, Action, Result) - analyze each project description and reformat to highlight the situation faced, tasks undertaken, actions taken, and measurable results achieved
            FORMATTING REQUIREMENTS:
        - Use **BOLD** formatting for ALL section headers (e.g., **PROFESSIONAL SUMMARY**, **CORE COMPETENCIES**, **PROFESSIONAL EXPERIENCE**, **PROJECTS**, **EDUCATION**)
        - Use **BOLD** formatting for job titles, company names, project titles, and degree titles
        - Use **BOLD** formatting for key achievement metrics and important skills
        - Section headers should be in ALL CAPS and bold: **SECTION NAME**
        
        IMPORTANT PROJECTS FORMATTING:
        For each project, transform the raw description into STAR format (in the form of a paragraph) where ONLY the project title is bold:
        - **Project Name**: Description using STAR method
        - Situation: What was the context or challenge? (NOT bold)
        - Task: What needed to be accomplished? (NOT bold)
        - Action: What specific actions did you take? (NOT bold)
        - Result: What measurable outcomes were achieved? (NOT bold)
        
        Do NOT make the words "Situation:", "Task:", "Action:", "Result:" bold.
        
        Format for both ATS and human readability, ensuring technical terms are used correctly.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert resume writer who creates compelling, ATS-optimized resumes with quantified achievements and current industry knowledge."},
            {"role": "user", "content": prompt}
        ]
        
        return self._make_request(messages, max_tokens=2000, temperature=0.6)

    def generate_tailored_resume(self, user_data: Dict[str, Any], job_description: str) -> str:
        search_results = self.search_unknown_terms(job_description, "job requirements career skills")
        projects = user_data.get('projects', [])
        
        enhanced_user_data = user_data.copy()
        if search_results:
            search_context = "\n\nResearched job context:\n"
            for term, explanation in search_results.items():
                search_context += f"- {term}: {explanation}\n"
            enhanced_job_description = job_description + search_context
        else:
            enhanced_job_description = job_description
        
        if projects:
            enhanced_user_data['projects'] = projects
            
        return self.generate_resume(enhanced_user_data, enhanced_job_description)

    def generate_resume(self, user_data: Dict[str, Any], job_description: str = "") -> str:
        tailoring_context = f"\n\nTailor the resume for this job:\n{job_description}" if job_description else ""
        projects = user_data.get('projects', [])
        
        projects_context = ""
        if projects:
            projects_context = "\n\nProjects to include (format using STAR method):\n"
            for i, project in enumerate(projects, 1):
                projects_context += f"\nProject {i}:\n"
                projects_context += f"- Title: {project.get('title', 'Untitled Project')}\n"
                projects_context += f"- Description: {project.get('description', 'No description provided')}\n"
                if project.get('technologies'):
                    projects_context += f"- Technologies: {project.get('technologies')}\n"
                if project.get('duration'):
                    projects_context += f"- Duration: {project.get('duration')}\n"
        
        prompt = f"""
        Create an ATS-optimized resume for:
        
        Name: {user_data.get('name') or '[Name to be provided]'}
        Title: {user_data.get('title') or '[Professional title to be specified]'}
        Email: {user_data.get('email') or '[Email address to be provided]'}
        Phone: {user_data.get('phone') or '[Phone number to be provided]'}
        Skills: {', '.join(user_data.get('skills', [])) if user_data.get('skills') else '[Skills to be specified]'}
        Experience: {user_data.get('experience') or '[Professional experience to be detailed]'}
        Education: {user_data.get('education') or '[Educational background to be provided]'}
        {tailoring_context}
        {projects_context}
        Create a professional, ATS-friendly resume with:
        1. Contact information
        2. Professional summary (3-4 lines)
        3. Core competencies/skills
        4. Professional experience with bullet points
        5. Projects section (if projects provided) - Format each project using STAR method
        6. Education
        7. Use action verbs and quantify achievements where possible        8. Include relevant keywords for ATS optimization
        
        FORMATTING REQUIREMENTS:
        - Use **BOLD** formatting for ALL section headers (e.g., **PROFESSIONAL SUMMARY**, **CORE COMPETENCIES**, **PROFESSIONAL EXPERIENCE**, **PROJECTS**, **EDUCATION**)
        - Use **BOLD** formatting for job titles, company names, project titles, and degree titles
        - Use **BOLD** formatting for key achievement metrics and important skills
        - Section headers should be in ALL CAPS and bold: **SECTION NAME**
        
        IMPORTANT PROJECTS FORMATTING:
        If projects are provided, create a dedicated "PROJECTS" section and format each project using the STAR method where ONLY the project title is bold:
        - **Project Name**: Description using STAR method
        - Situation: What was the context or challenge? (NOT bold)
        - Task: What needed to be accomplished? (NOT bold)
        - Action: What specific actions did you take? (NOT bold)
        - Result: What measurable outcomes were achieved? (NOT bold)
        
        Do NOT make the words "Situation:", "Task:", "Action:", "Result:" bold. Only the project title should be bold.
        
        Transform the raw project descriptions into compelling STAR-formatted entries that showcase impact and achievements.
        
        Format in clean, readable text suitable for both ATS and human review.
        """
        
        messages = [
            {
                "role": "system", 
                "content": "You are an expert resume writer specializing in ATS-optimized resumes that get results."
            },
            {"role": "user", "content": prompt}
        ]
        
        return self._make_request(messages, max_tokens=2000, temperature=0.6)

    def generate_interview_questions(self, job_description: str, user_data: Dict[str, Any], num_questions: int = 5) -> List[Dict]:
        name = str(user_data.get('name', 'Candidate'))
        title = str(user_data.get('title', 'Professional'))
        skills = user_data.get('skills', [])
        if isinstance(skills, list):
            skills_str = ', '.join(str(skill) for skill in skills)
        else:
            skills_str = str(skills) if skills else 'Various skills'
        experience = str(user_data.get('experience', 'Professional experience'))
        prompt = f"""
        Generate {num_questions} interview questions for this position:
        
        Job Description:
        {job_description}
        
        Candidate Profile:
        Name: {name}
        Title: {title}
        Skills: {skills_str}
        Experience: {experience}
        
        Create a mix of:
        1. Behavioral questions (STAR method)
        2. Technical questions (based on skills)
        3. Situational questions
        4. Company/role-specific questions
        
        Return as JSON array:
        [
            {{
                "question": "Tell me about yourself",
                "type": "General",
                "difficulty": "Easy",
                "category": "Introduction"
            }},
            ...
        ]
        """
        messages = [
            {"role": "system", "content": "You are an expert interview coach who creates thoughtful, relevant interview questions. Return only valid JSON array."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = self._make_request(messages, max_tokens=1500, temperature=0.7)
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start != -1 and json_end != 0:
                json_str = response[json_start:json_end]
                questions = json.loads(json_str)
                if isinstance(questions, list):
                    # Always return exactly num_questions
                    if len(questions) > num_questions:
                        return questions[:num_questions]
                    elif len(questions) < num_questions:
                        # Pad with generic questions if too few
                        default_qs = [
                            {"question": "Tell me about yourself and your background.", "type": "General", "difficulty": "Easy", "category": "Introduction"},
                            {"question": "Why are you interested in this position?", "type": "Behavioral", "difficulty": "Easy", "category": "Motivation"},
                            {"question": "Describe a challenging project you worked on and how you handled it.", "type": "Behavioral", "difficulty": "Medium", "category": "Problem Solving"},
                            {"question": "What are your greatest strengths and how do they apply to this role?", "type": "General", "difficulty": "Medium", "category": "Self Assessment"},
                            {"question": "Where do you see yourself in 5 years?", "type": "General", "difficulty": "Medium", "category": "Career Goals"}
                        ]
                        while len(questions) < num_questions:
                            questions.append(default_qs[len(questions) % len(default_qs)])
                        return questions
                    else:
                        return questions
        except Exception as e:
            print(f"Error generating interview questions: {e}")
        # Fallback: always return exactly num_questions
        default_qs = [
            {"question": "Tell me about yourself and your background.", "type": "General", "difficulty": "Easy", "category": "Introduction"},
            {"question": "Why are you interested in this position?", "type": "Behavioral", "difficulty": "Easy", "category": "Motivation"},
            {"question": "Describe a challenging project you worked on and how you handled it.", "type": "Behavioral", "difficulty": "Medium", "category": "Problem Solving"},
            {"question": "What are your greatest strengths and how do they apply to this role?", "type": "General", "difficulty": "Medium", "category": "Self Assessment"},
            {"question": "Where do you see yourself in 5 years?", "type": "General", "difficulty": "Medium", "category": "Career Goals"}
        ]
        return [default_qs[i % len(default_qs)] for i in range(num_questions)]

    def generate_interview_question(self, user_data: Dict[str, Any], job_description: str = "") -> str:
        questions = self.generate_interview_questions(job_description, user_data, 1)
        return questions[0]['question'] if questions else "Tell me about your experience and background."

    def evaluate_interview_answer(self, question: str, answer: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        skills = user_data.get('skills', [])
        if isinstance(skills, list):
            skills_str = ', '.join(str(skill) for skill in skills)
        else:
            skills_str = str(skills) if skills else 'Not specified'
        experience = str(user_data.get('experience', 'Professional experience'))
        question_safe = str(question).replace('{', '{{').replace('}', '}}')
        answer_safe = str(answer).replace('{', '{{').replace('}', '}}')

        prompt = f"""
        Evaluate this interview answer:
        
        Question: {question_safe}
        Answer: {answer_safe}
        
        Candidate Profile:
        Skills: {skills_str}
        Experience: {experience}
        
        Provide evaluation with:
        1. Score (1-10)
        2. Strengths in the answer
        3. Areas for improvement
        4. Specific suggestions
        
        Return as JSON:
        {{
            "score": 7,
            "strengths": ["Clear communication", "Relevant example"],
            "weaknesses": ["Could be more specific", "Missing quantified results"],
            "suggestions": "Try to include specific metrics and outcomes in your examples.",
            "feedback": "Good response overall, but could be enhanced with more concrete details."
        }}
        """

        messages = [
            {"role": "system", "content": "You are an expert interview coach providing constructive feedback. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self._make_request(messages, max_tokens=800, temperature=0.6)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except Exception as e:
            print(f"Error in evaluate_interview_answer: {e}")

        # Fallback: If the answer is very short or obviously bad, give a low score
        answer_stripped = str(answer).strip().lower()
        bad_phrases = ["i don't know", "no idea", "not sure", "n/a", "none", "idk", "?", "", "-", "skip", "nothing"]
        if len(answer_stripped) < 10 or any(p in answer_stripped for p in bad_phrases):
            return {
                "score": 2,
                "strengths": [],
                "weaknesses": ["Answer is too short or not relevant."],
                "suggestions": "Try to provide a more complete and relevant answer.",
                "feedback": "Your answer was too brief or not relevant. Please elaborate and provide examples next time."
            }
        return {
            "score": 6,
            "strengths": ["Good effort"],
            "weaknesses": ["Could provide more detail"],
            "suggestions": "Consider using the STAR method for behavioral questions.",
            "feedback": "Thank you for your response. Consider adding more specific examples."
        }

    def generate_chat_interview_question(self, context: str, user_data: Dict[str, Any], question_number: int) -> str:
        skills = user_data.get('skills', [])
        if isinstance(skills, list):
            skills_str = ', '.join(str(skill) for skill in skills)
        else:
            skills_str = str(skills) if skills else 'Various skills'
        experience = str(user_data.get('experience', 'Professional experience'))
        
        prompt = f"""
        You are conducting a conversational job interview. Generate the next question based on:
        
        {context}
        
        Candidate Background:
        Skills: {skills_str}
        Experience: {experience}
        
        This is question #{question_number} of 5. Generate a natural, conversational interview question that:
        1. Flows from the previous conversation
        2. Is appropriate for the role and experience level
        3. Allows the candidate to showcase their skills
        4. Feels like a real interview conversation
        
        Return only the question text, no additional formatting.
        """
        
        messages = [
            {"role": "system", "content": "You are a professional interviewer conducting a conversational job interview. Ask thoughtful, relevant questions."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self._make_request(messages, max_tokens=200, temperature=0.7)
            return response if response and not response.startswith("❌") else "Can you tell me more about your relevant experience for this role?"
        except Exception as e:
            print(f"Error generating chat interview question: {e}")
            return "Can you tell me more about your relevant experience for this role?"

    def analyze_chat_interview(self, questions: List[str], answers: List[str], job_info: Dict[str, Any], user_data: Dict[str, Any]) -> Dict[str, Any]:
        interview_content = ""
        for i, (q, a) in enumerate(zip(questions, answers), 1):
            interview_content += f"Q{i}: {q}\nA{i}: {a}\n\n"
        
        # Use triple curly braces to escape braces in the JSON example for f-string
        prompt = f"""
        Analyze this complete job interview performance:
        
        Job Information:
        Position: {job_info.get('job_title', 'N/A')}
        Company: {job_info.get('company', 'N/A')}
        Experience Level: {job_info.get('experience_level', 'N/A')}
        Interview Type: {job_info.get('interview_type', 'N/A')}
        
        Interview Conversation:
        {interview_content}
        
        Candidate Profile:
        Skills: {', '.join(user_data.get('skills', []))}
        Experience: {user_data.get('experience', 'Professional experience')}
        
        Provide comprehensive analysis:
        {{
            "overall_score": 8.5,
            "performance_level": "Excellent",
            "strengths": ["Strong communication", "Relevant examples", "Technical knowledge"],
            "improvement_areas": ["Could provide more specific metrics", "Expand on leadership examples"],
            "detailed_feedback": "The candidate demonstrated strong technical knowledge and communication skills...",
            "question_scores": [8, 7, 9, 8, 7],
            "recommendations": ["Practice quantifying achievements", "Prepare more leadership stories"]
        }}
        """
        messages = [
            {"role": "system", "content": "You are an expert interview analyst providing detailed performance feedback. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ]
        response = self._make_request(messages, max_tokens=1500, temperature=0.6)
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except:
            return {
                "overall_score": 7.5,
                "performance_level": "Good",
                "strengths": ["Good communication", "Relevant experience"],
                "improvement_areas": ["Provide more specific examples", "Quantify achievements"],
                "detailed_feedback": "You demonstrated good knowledge and communication skills. Consider adding more specific examples and quantifiable achievements in future interviews.",
                "question_scores": [7] * len(questions),
                "recommendations": ["Practice the STAR method", "Prepare specific achievement stories"]
            }

    def analyze_job_matches(self, jobs: List[Dict[str, Any]], user_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        enhanced_jobs = []
        
        for job in jobs:
            try:
                job_desc = job.get('description', '')
                job_title = job.get('title', '')
                
                prompt = f"""
                Analyze this job posting against the candidate's profile:
                
                Job Title: {job_title}
                Job Description: {job_desc}
                
                Candidate Profile:
                Skills: {', '.join(user_data.get('skills', []))}
                Experience: {user_data.get('experience', 'Professional experience')}
                Title: {user_data.get('title', 'Professional')}
                
                Provide a match analysis:
                {{
                    "match_score": 85,
                    "match_level": "Excellent",
                    "matched_keywords": ["Python", "React", "API"],
                    "missing_skills": ["Docker", "AWS"],
                    "strengths": ["Strong technical background", "Relevant experience"],
                    "recommendations": ["Highlight Python experience", "Mention API development projects"]
                }}
                
                Return only valid JSON.
                """
                
                messages = [
                    {"role": "system", "content": "You are an expert job match analyzer. Return only valid JSON with match analysis."},
                    {"role": "user", "content": prompt}
                ]
                
                response = self._make_request(messages, max_tokens=500, temperature=0.3)
                
                try:
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start != -1 and json_end != 0:
                        json_str = response[json_start:json_end]
                        analysis = json.loads(json_str)
                        job['ai_analysis'] = analysis
                        job['ai_match_score'] = analysis.get('match_score', 75)
                    else:
                        job['ai_analysis'] = {
                            "match_score": 75,
                            "match_level": "Good",
                            "matched_keywords": [],
                            "missing_skills": [],
                            "strengths": ["Review job requirements"],
                            "recommendations": ["Tailor your application"]
                        }
                        job['ai_match_score'] = 75
                except:
                    job['ai_analysis'] = {
                        "match_score": 75,
                        "match_level": "Good",
                        "matched_keywords": [],
                        "missing_skills": [],
                        "strengths": ["Review job requirements"],
                        "recommendations": ["Tailor your application"]
                    }
                    job['ai_match_score'] = 75
                
            except Exception as e:
                job['ai_analysis'] = {
                    "match_score": 70,
                    "match_level": "Fair",
                    "matched_keywords": [],
                    "missing_skills": [],
                    "strengths": ["Review carefully"],
                    "recommendations": ["Analyze job requirements"]
                }
                job['ai_match_score'] = 70
            
            enhanced_jobs.append(job)
        
        enhanced_jobs.sort(key=lambda x: x.get('ai_match_score', 0), reverse=True)
        return enhanced_jobs

    def chat_about_resume(self, resume_content: str, user_message: str, chat_history: List[Dict] = None) -> str:
        if chat_history is None:
            chat_history = []
        
        conversation = ""
        for msg in chat_history[-5:]: 
            role = "You" if msg['role'] == 'user' else "AI Assistant"
            conversation += f"{role}: {msg['content']}\n"
        
        prompt = f"""
        You are an expert career counselor and resume specialist. The user has uploaded their resume and wants to discuss it with you.
        
        Resume Content:
        {resume_content}
        
        Previous Conversation:
        {conversation}
        
        User's Question/Message:
        {user_message}
        
        Provide helpful, specific advice about their resume. You can:
        1. Answer questions about resume content
        2. Suggest improvements to specific sections
        3. Analyze strengths and weaknesses
        4. Provide industry-specific advice
        5. Help optimize for ATS systems
        6. Suggest additional skills or experiences to highlight
        7. Help tailor the resume for specific jobs
        Be conversational, supportive, and provide actionable advice.
        """
        messages = [
            {"role": "system", "content": "You are a friendly and expert career counselor who helps people improve their resumes. Be conversational, supportive, and provide specific, actionable advice."},
            {"role": "user", "content": prompt}
        ]
        
        return self._make_request(messages, max_tokens=1000, temperature=0.7)

    def chat_with_resume(self, user_message: str, context: str) -> str:
        try:
            return self.chat_about_resume(context, user_message, chat_history=None)
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your request: {str(e)}. Please try asking your question in a different way."

    def analyze_job_requirements(self, job_description: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
        Analyze this job description against the candidate's profile:
        
        Job Description:
        {job_description}
        
        Candidate Profile:
        Skills: {', '.join(user_data.get('skills', []))}
        Experience: {user_data.get('experience', 'Professional experience')}
        Title: {user_data.get('title', 'Professional')}
        
        Provide analysis:
        {
            "match_percentage": 85,
            "keyword_matches": 12,
            "missing_skills": ["Python", "Docker"],
            "matching_skills": ["JavaScript", "React", "Node.js"],
            "recommendations": ["Highlight your JavaScript experience", "Consider learning Python"]
        }
        """
        
        messages = [
            {"role": "system", "content": "You are an expert job match analyzer. Return only valid JSON."},
            {"role": "user", "content": prompt}        ]
        
        response = self._make_request(messages, max_tokens=800, temperature=0.3)
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except:
            return {
                "match_percentage": 75,
                "keyword_matches": 8,
                "missing_skills": ["Review job requirements"],
                "matching_skills": user_data.get('skills', [])[:5],
                "recommendations": ["Tailor your resume to match job requirements"]
            }

    def parse_resume_data(self, resume_text: str) -> Dict[str, Any]:
        prompt = f"""
        As an expert resume analyst, perform a comprehensive analysis and extraction of ALL information from this resume text. Extract every detail accurately and completely.

        Resume Text:
        {resume_text}

        Analyze the resume text thoroughly and extract ALL available information. Return a complete JSON structure:
        {{
            "name": "Full Name (extract from header/contact section)",
            "email": "email@example.com",
            "phone": "Phone Number with formatting",
            "title": "Current or target job title/professional title",
            "location": "City, State/Country if found",
            "linkedin": "LinkedIn URL if found",
            "website": "Personal website/portfolio URL if found",
            "summary": "Professional summary or objective statement if present",
            "skills": ["comprehensive list of ALL skills mentioned including technical, soft, programming languages, frameworks, tools, certifications"],
            "experience": [
                {{
                    "company": "Company Name",
                    "position": "Job Title", 
                    "duration": "Start Date - End Date",
                    "location": "Work location if mentioned",
                    "description": "Detailed job description with achievements and responsibilities",
                    "achievements": ["List of specific achievements with metrics if available"]
                }}
            ],
            "education": [
                {{
                    "institution": "University/School Name",
                    "degree": "Degree Type and Major",
                    "graduation_year": "Year or date",
                    "gpa": "GPA if mentioned",
                    "location": "Location if mentioned",
                    "relevant_coursework": "Relevant courses if mentioned"
                }}
            ],
            "projects": [
                {{
                    "title": "Project Name",
                    "description": "Comprehensive description of what was built/achieved",
                    "technologies": "All technologies, frameworks, languages used",
                    "duration": "Timeline or duration",
                    "role": "Your role in the project",
                    "achievements": "Key outcomes, metrics, impact",
                    "links": "GitHub, demo, or project links if mentioned"
                }}
            ],
            "certifications": ["List of certifications with issuing organizations"],
            "languages": ["Spoken languages with proficiency levels if mentioned"],
            "publications": ["Research papers, articles, publications if any"],
            "awards": ["Awards, honors, recognitions if any"],
            "volunteer_experience": ["Volunteer work or community involvement"],
            "additional_sections": {{
                "interests": ["Hobbies and interests"],
                "references": "Reference information if provided"
            }}
        }}

        CRITICAL EXTRACTION GUIDELINES:

        1. **PROJECTS - Extract EVERYTHING that resembles a project:**
        - Dedicated project sections
        - Work projects mentioned in job descriptions
        - Academic/thesis projects
        - Personal/side projects
        - Open source contributions
        - Hackathon projects
        - Portfolio pieces
        - Apps, websites, tools, software developed
        - Research projects
        - Capstone projects
        - Freelance work
        - ANY mentions of building, creating, developing, designing, implementing

        2. **SKILLS - Be exhaustive:**
        - Programming languages (Python, Java, JavaScript, etc.)
        - Frameworks and libraries (React, Django, TensorFlow, etc.)
        - Tools and software (Git, Docker, AWS, Figma, etc.)
        - Technical skills (Machine Learning, Data Analysis, etc.)
        - Soft skills (Leadership, Communication, etc.)
        - Industry-specific skills
        - Certifications and qualifications

        3. **EXPERIENCE - Extract complete work history:**
        - Include internships, part-time jobs, freelance work
        - Extract specific achievements with numbers/metrics
        - Include all responsibilities and accomplishments

        4. **EDUCATION - Complete academic background:**
        - All degrees, diplomas, certificates
        - Relevant coursework, thesis topics
        - Academic achievements, GPA, honors

        5. **Contact Information - Extract all available:**
        - Full name, professional email, phone
        - LinkedIn profile, personal website, GitHub
        - Location/address information

        6. **Quality Standards:**
        - Extract exact text, don't paraphrase
        - Include metrics, percentages, dollar amounts
        - Preserve technical terminology
        - Use "Not found" only if truly absent
        - Ensure valid JSON with proper escaping

        Analyze every line of the resume. Don't miss any information that could be valuable for career development.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert resume parser and career analyst. Perform comprehensive extraction of ALL resume information. Return only valid, complete JSON."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._make_request(messages, max_tokens=2500, temperature=0.2)
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = response[json_start:json_end]
                parsed_data = json.loads(json_str)
                parsed_data = self._validate_and_enhance_parsed_data(parsed_data)
                
                return parsed_data
        except Exception as e:
            print(f"Error parsing resume data with enhanced LLM: {e}")
            return self._fallback_resume_parsing(resume_text)
    
    def _validate_and_enhance_parsed_data(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            required_fields = {
                'name': 'Not found',
                'email': 'Not found',
                'phone': 'Not found',
                'title': 'Not found',
                'location': 'Not found',
                'linkedin': 'Not found',
                'website': 'Not found',
                'summary': 'Not found',
                'skills': [],
                'experience': [],
                'education': [],
                'projects': [],
                'certifications': [],
                'languages': [],
                'publications': [],
                'awards': [],
                'volunteer_experience': [],
                'additional_sections': {}
            }
            
            for field, default_value in required_fields.items():
                if field not in parsed_data or parsed_data[field] is None:
                    parsed_data[field] = default_value
            
            if isinstance(parsed_data.get('skills'), str):
                parsed_data['skills'] = [skill.strip() for skill in parsed_data['skills'].split(',') if skill.strip()]
            elif not isinstance(parsed_data.get('skills'), list):
                parsed_data['skills'] = []
            
            parsed_data['skills'] = list(set([skill for skill in parsed_data['skills'] if skill and isinstance(skill, str) and len(skill.strip()) > 1]))
            
            if not isinstance(parsed_data.get('experience'), list):
                exp_text = str(parsed_data.get('experience', ''))
                if exp_text and exp_text != 'Not found':
                    parsed_data['experience'] = [{
                        'company': 'Previous Experience',
                        'position': 'Professional',
                        'duration': 'Previous role',
                        'description': exp_text[:300] + '...' if len(exp_text) > 300 else exp_text,
                        'achievements': []
                    }]
                else:
                    parsed_data['experience'] = []
            
            if not isinstance(parsed_data.get('education'), list):
                edu_text = str(parsed_data.get('education', ''))
                if edu_text and edu_text != 'Not found':
                    parsed_data['education'] = [{
                        'institution': 'Educational Institution',
                        'degree': edu_text[:100] + '...' if len(edu_text) > 100 else edu_text,
                        'graduation_year': 'Not specified',
                        'location': 'Not specified'
                    }]
                else:
                    parsed_data['education'] = []
            
            if not isinstance(parsed_data.get('projects'), list):
                parsed_data['projects'] = []
            
            clean_projects = []
            for project in parsed_data.get('projects', []):
                if isinstance(project, dict) and project.get('title'):
                    clean_project = {
                        'title': str(project.get('title', 'Untitled Project'))[:100],
                        'description': str(project.get('description', 'Project details available'))[:500],
                        'technologies': str(project.get('technologies', 'Not specified'))[:200],
                        'duration': str(project.get('duration', 'Not specified'))[:50],
                        'role': str(project.get('role', 'Team Member'))[:100],
                        'achievements': str(project.get('achievements', 'Successful completion'))[:300],
                        'links': str(project.get('links', 'Not available'))[:200]
                    }
                    clean_projects.append(clean_project)
            
            parsed_data['projects'] = clean_projects[:10]  
            
            string_fields = ['name', 'email', 'phone', 'title', 'location', 'linkedin', 'website', 'summary']
            for field in string_fields:
                if isinstance(parsed_data.get(field), str):
                    parsed_data[field] = parsed_data[field].strip()
                    if len(parsed_data[field]) > 500: 
                        parsed_data[field] = parsed_data[field][:500] + '...'
            
            list_fields = ['certifications', 'languages', 'publications', 'awards', 'volunteer_experience']
            for field in list_fields:
                if not isinstance(parsed_data.get(field), list):
                    parsed_data[field] = []
                parsed_data[field] = [str(item)[:200] for item in parsed_data[field] if item][:10]
            
            if not isinstance(parsed_data.get('additional_sections'), dict):
                parsed_data['additional_sections'] = {}
            
            return parsed_data
            
        except Exception as e:
            print(f"Error validating parsed data: {e}")
            return {
                'name': 'Error in processing',
                'email': 'Not found',
                'phone': 'Not found',
                'title': 'Not found',
                'skills': [],
                'experience': [],
                'education': [],
                'projects': []
            }
    
    def _fallback_resume_parsing(self, resume_text: str) -> Dict[str, Any]:
        import re
        
        parsed_data = {
            "name": "Not found",
            "email": "Not found", 
            "phone": "Not found",
            "title": "Not found",
            "skills": [],
            "experience": "Not found",
            "education": "Not found",
            "linkedin": "Not found",
            "location": "Not found",
            "projects": []
        }
        
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, resume_text)
        if email_match:
            parsed_data["email"] = email_match.group()
        
        phone_pattern = r'(\+?1?[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phone_match = re.search(phone_pattern, resume_text)
        if phone_match:
            parsed_data["phone"] = phone_match.group()
        
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin_match = re.search(linkedin_pattern, resume_text, re.IGNORECASE)
        if linkedin_match:
            parsed_data["linkedin"] = linkedin_match.group()
        
        lines = resume_text.split('\n')
        for line in lines[:5]:  
            line = line.strip()
            if line and len(line) > 3 and len(line) < 50:
                if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+', line):
                    parsed_data["name"] = line
                    break
        
        experience_keywords = ['experience', 'work', 'employment', 'career', 'professional']
        for keyword in experience_keywords:
            pattern = rf'{keyword}.*?(?=\n\n|\Z)'
            match = re.search(pattern, resume_text, re.IGNORECASE | re.DOTALL)
            if match:
                parsed_data["experience"] = match.group()[:200] + "..."
                break
        project_sections = ['project', 'portfolio', 'work', 'experience', 'academic', 'capstone', 'research']
        project_keywords = [
            'built', 'developed', 'created', 'designed', 'implemented', 'programmed',
            'website', 'application', 'app', 'system', 'tool', 'platform', 'dashboard',
            'analysis', 'model', 'algorithm', 'database', 'api', 'software', 'solution'
        ]
        projects = []
        
        text_lines = resume_text.split('\n')
        current_section = ""
        
        for i, line in enumerate(text_lines):
            line = line.strip()
            
            if any(section in line.lower() for section in project_sections) and len(line) < 50:
                current_section = line.lower()
                continue
            
            if any(keyword in line.lower() for keyword in project_keywords):
                if len(line) > 10 and len(line) < 150:  
                    title = line
                    if title.startswith('•') or title.startswith('-') or title.startswith('*'):
                        title = title[1:].strip()
                    
                    description_parts = []
                    
                    for j in range(i, min(i+4, len(text_lines))):
                        desc_line = text_lines[j].strip()
                        if desc_line and len(desc_line) > 15:
                            if desc_line.startswith('•') or desc_line.startswith('-') or desc_line.startswith('*'):
                                desc_line = desc_line[1:].strip()
                            
                            if any(tech_word in desc_line.lower() for tech_word in ['using', 'with', 'implemented', 'achieved', 'resulted']):
                                description_parts.append(desc_line)
                    
                    description = " ".join(description_parts[:2]) if description_parts else title
                    
                    technologies = []
                    tech_keywords = [
                        'python', 'javascript', 'java', 'react', 'node', 'html', 'css', 'sql',
                        'mongodb', 'postgresql', 'django', 'flask', 'express', 'angular', 'vue',
                        'git', 'docker', 'aws', 'azure', 'gcp', 'tensorflow', 'pytorch', 'pandas',
                        'numpy', 'scikit', 'tableau', 'powerbi', 'excel', 'r', 'matlab', 'c++', 'c#'
                    ]
                    
                    search_text = " ".join(text_lines[max(0, i-1):min(len(text_lines), i+3)]).lower()
                    for tech in tech_keywords:
                        if tech in search_text:
                            technologies.append(tech.capitalize())
                    
                    duration = "Not specified"
                    duration_patterns = [
                        r'\d+\s*months?', r'\d+\s*weeks?', r'\d+\s*years?',
                        r'(spring|summer|fall|winter)\s*\d{4}',
                        r'\d{4}\s*-\s*\d{4}', r'\d{1,2}/\d{4}\s*-\s*\d{1,2}/\d{4}'
                    ]
                    for pattern in duration_patterns:
                        match = re.search(pattern, search_text, re.IGNORECASE)
                        if match:
                            duration = match.group()
                            break
                    
                    if len(title) > 5 and not title.lower().startswith('experience'):
                        projects.append({
                            "title": title[:100],
                            "description": description[:300] if description else "Project details available upon request",
                            "technologies": ", ".join(technologies[:5]) if technologies else "Not specified",                            "duration": duration
                        })
        
        unique_projects = []
        seen_titles = set()
        for project in projects:
            title_lower = project['title'].lower()
            if title_lower not in seen_titles and len(title_lower) > 3:
                seen_titles.add(title_lower)
                unique_projects.append(project)
                if len(unique_projects) >= 5:
                    break
        
        parsed_data["projects"] = unique_projects
        
        return parsed_data
    
    def generate_enhanced_portfolio(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        include_projects = user_data.get('include_projects', False)
        
        if include_projects:
            projects_instruction = '''
            "projects": [
                {
                    "name": "Project Name",
                    "description": "Brief project description highlighting achievements",
                    "technologies": ["tech1", "tech2"]
                }
            ],'''
        else:
            projects_instruction = '''
            "projects": [],'''
            
        prompt = f"""
        Create a comprehensive and professional portfolio content for the following person.
        Make it engaging, well-structured, and highlight their strengths and achievements.
        IMPORTANT: Use the actual user data provided below. Only enhance the content, don't replace real information.
        
        User Data:
        Name: {user_data.get('name', 'N/A')}
        Email: {user_data.get('email', 'N/A')}
        Phone: {user_data.get('phone', 'N/A')}
        LinkedIn: {user_data.get('linkedin', 'N/A')}
        GitHub: {user_data.get('github', 'N/A')}
        Summary: {user_data.get('summary', 'N/A')}
        Skills: {user_data.get('skills', 'N/A')}
        Experience: {user_data.get('experience', 'N/A')}
        Education: {user_data.get('education', 'N/A')}
        Projects: {user_data.get('projects', 'N/A')}
        Certifications: {user_data.get('certifications', 'N/A')}
        
        Return a structured JSON portfolio with these sections:
        {{
            "headline": "Professional headline/tagline",
            "about": "Compelling professional summary",
            "skills": ["skill1", "skill2", "skill3"],{projects_instruction}
            "experience": [
                {{
                    "title": "Job Title",
                    "company": "Company Name",
                    "duration": "Date Range",
                    "achievements": ["Achievement 1", "Achievement 2"]
                }}
            ],
            "education": "Education summary",
            "certifications": ["cert1", "cert2"]
        }}
        
        Make it professional, compelling, and tailored to showcase their unique value proposition.
        Use modern, engaging language and highlight quantifiable achievements where possible.
        {"If include_projects is False, do NOT generate any projects - return an empty projects array." if not include_projects else ""}        Return only valid JSON.        """
        
        messages = [
            {"role": "system", "content": "You are a professional portfolio writer. Create structured portfolio data in JSON format only."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self._make_request(messages, max_tokens=2500, temperature=0.8)
            
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = response[json_start:json_end]
                json_str = json_str.strip()
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                json_str = json_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                
                try:
                    portfolio_data = json.loads(json_str)
                    
                    if not isinstance(portfolio_data, dict):
                        raise ValueError("Portfolio data is not a dictionary")
                    
                    portfolio_data.setdefault('headline', f"{user_data.get('title', 'Professional')} | Technology Expert")
                    portfolio_data.setdefault('about', user_data.get('summary', 'Experienced professional with a passion for innovation'))
                    portfolio_data.setdefault('skills', user_data.get('skills', []))
                    
                    if not user_data.get('include_projects', False):
                        portfolio_data['projects'] = []
                    else:
                        portfolio_data.setdefault('projects', [])
                    if user_data.get('work_experience'):
                        portfolio_data['experience'] = user_data['work_experience']
                    elif user_data.get('experience'):
                        portfolio_data.setdefault('experience', [{
                            'title': user_data.get('title', 'Professional'),
                            'company': 'Professional Experience',
                            'duration': 'Current',
                            'description': user_data.get('experience')
                        }])
                    else:
                        portfolio_data.setdefault('experience', [])
                    
                    portfolio_data.setdefault('education', user_data.get('education', ''))
                    portfolio_data.setdefault('certifications', user_data.get('certifications', []))
                    
                    return portfolio_data
                    
                except json.JSONDecodeError as json_error:
                    print(f"JSON parsing error: {str(json_error)}")
                    print(f"Problematic JSON: {json_str[:500]}...")
                    return self._create_fallback_portfolio(user_data)
            else:
                print("No valid JSON found in response")
                return self._create_fallback_portfolio(user_data)
                
        except Exception as e:
            print(f"Error generating enhanced portfolio: {str(e)}")
            return self._create_fallback_portfolio(user_data)
    
    def _create_fallback_portfolio(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        projects = []
        if user_data.get('include_projects', False):
            projects = [
                {
                    "name": "Professional Project Portfolio",
                    "description": "Comprehensive collection of professional projects demonstrating technical expertise and problem-solving capabilities",
                    "technologies": user_data.get('skills', ['Various Technologies'])[:3]
                }
            ]
        
        experience = []
        if user_data.get('work_experience'):
            experience = user_data['work_experience']
        elif user_data.get('experience'):
            experience = [{
                "title": user_data.get('title', 'Professional'),
                "company": "Professional Experience",
                "duration": "Current",
                "description": user_data.get('experience')
            }]
        else:
            experience = [{
                "title": user_data.get('title', 'Professional'),
                "company": "Professional Experience",
                "duration": "Current",
                "achievements": ["Delivered high-quality solutions", "Collaborated with cross-functional teams", "Contributed to organizational success"]
            }]
        
        return {
            "headline": f"{user_data.get('title', 'Professional')} | Experienced in Technology & Innovation",
            "about": user_data.get('summary', f"Accomplished {user_data.get('title', 'professional')} with extensive experience in delivering high-quality solutions. Passionate about technology and innovation with a proven track record of success."),
            "skills": user_data.get('skills', ['Leadership', 'Problem Solving', 'Communication', 'Technical Skills']),
            "projects": projects,
            "experience": experience,
            "education": user_data.get('education', 'Educational background in relevant field'),
            "certifications": user_data.get('certifications', ['Professional Certifications'])
        }
