import os
import streamlit as st
from openai import OpenAI
from langchain_community.llms import Ollama

# --- CORE AGENT ENGINE ---
class TestGeneratorAgent:
    def __init__(self, provider, model_name, api_key=None, base_url=None):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    def _generate_prompt(self, target_lang: str, code_content: str) -> str:
        is_csharp = target_lang.strip().lower() in ("c#", "csharp")
        framework = "NUnit with Moq only when needed" if is_csharp else "Jest/Vitest"

        naming_rules = """
    For C#, every test method must follow:
    <MethodName>_<Positive|Negative|Boundary>_<Input>_<ExpectedResult>

    Examples:
    Add_Positive_TwoPositiveIntegers_ReturnsSum
    Add_Negative_NegativeAndPositiveIntegers_ReturnsSum
    Add_Boundary_ZeroAndZero_ReturnsZero

    Use PascalCase segments separated by underscores.
    """ if is_csharp else """
    For TypeScript, preserve the existing Jest/Vitest framework and project
    naming conventions. Use descriptive test names for each scenario.
    """

        return f"""
    Ignore all previous instructions, history, and assumptions. Treat this as a completely independent task. Use ONLY the source code and requirements provided below.

CRITICAL RULES (read carefully):
- Do NOT invent any classes, interfaces, constructors, services, or dependencies.
- Do NOT create a test class, [TestFixture], [SetUp], or any fixture when the input is a snippet.
- Do NOT use Moq / Mock when the input is a simple method.
- Do NOT mock the method under test under any circumstances.
- Call the real method directly by its name.

If the input is a simple method like:
public int Add(int a, int b) {{ return a + b; }}

Then the ONLY correct output is pure test methods like this:

[Test]
public void Add_Positive_TwoPositiveIntegers_ReturnsSum()
{{
    int result = Add(2, 3);
    Assert.That(result, Is.EqualTo(5));
}}

Any output that contains `using`, `[TestFixture]`, `Mock<>`, invented interfaces (`IAdd`, `IAdditionService`, etc.), or a test class is WRONG.

You are a senior QA Automation Engineer specializing in {target_lang}.
Your goal is to produce meaningful, maintainable unit tests that maximize SonarQube line and branch coverage.

Target language: {target_lang}
Testing framework: {framework}

SOURCE CODE:
{code_content}

### 1. Determine Input Type
- **Snippet**: single method / property / function / partial class → output ONLY test methods.
- **Complete file**: full class with real dependencies → full test file is allowed.

### 2. Output Format

**CASE A — Snippet (strictly enforced)**
Output ONLY the test methods. Nothing else.

Forbidden in snippet mode (any of these make the answer invalid):
- using / import statements
- namespaces
- [TestFixture] or any class
- [SetUp] / [TearDown]
- Moq / Mock
- inventing interfaces or classes (IAdd, AddService, etc.)
- mocking the method under test
- Assert.AreEqual (prefer Assert.That)

Correct pattern:
[Test]
public void MethodName_Positive_..._Returns...()
{{
    var result = MethodName(...);
    Assert.That(result, Is.EqualTo(...));
}}

**CASE B — Complete Source File**
- Full compilable test file is allowed.
- Use only the real classes and constructors that exist in the source.
- Mock only real external dependencies that appear in the source.
- Never invent missing types.

### 3. Testing Requirements
- Cover positive and negative scenarios.
- Add boundary cases only when they exercise distinct behavior.
- Never invent failure scenarios the code cannot produce.
- Never mock the method under test.
- Never invent production code.
- Do not test private methods directly.
- Keep tests deterministic and independent.

### 4. C# Rules
- NUnit only (`[Test]`, `[TestCase]`).
- Prefer `Assert.That`.
- Moq only for real external dependencies that exist in the source (never for the method under test).
- Arrange-Act-Assert.
- Async → `async Task` + `await`.

### 5. TypeScript Rules
- Preserve existing Jest/Vitest style and conventions.
- Proper types, no unnecessary `any`.

### 6. Naming
{naming_rules}

### 7. Final Output Rules
- Return only raw source code. No explanations, no markdown.
- Snippet → pure test methods only.
- If you generate `using`, `[TestFixture]`, `Mock<>`, or any invented type for a snippet → the answer is invalid. Start over and output only the test methods.
    """


    def generate_tests(self, target_lang: str, code_content: str) -> str:
        prompt = self._generate_prompt(target_lang, code_content)
        
        # Scenario A: Anthropic Claude
        if self.provider == "Anthropic (Claude)":
            if not self.api_key: return "Error: Missing Anthropic API Key."
            try:
                client = Anthropic(api_key=self.api_key)
                response = client.messages.create(
                    model=self.model_name,
                    max_tokens=4096,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                # Handle text block serialization safely
                return response.content[0].text
            except Exception as e:
                return f"Anthropic Claude API Error: {str(e)}"

        # Scenario B: Cloud (OpenAI)
        elif self.provider == "Cloud (OpenAI)":
            if not self.api_key: return "Error: Missing Cloud API Key."
            try:
                client = OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                return response.choices.message.content
            except Exception as e:
                return f"Cloud LLM Error: {str(e)}"
        
        # Scenario C: DeepSeek
        elif self.provider == "DeepSeek":
            if not self.api_key: return "Error: Missing DeepSeek API Key."
            try:
                client = OpenAI(api_key=self.api_key, base_url="https://deepseek.com")
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                return response.choices.message.content
            except Exception as e:
                return f"DeepSeek API Error: {str(e)}"
        
        # Scenario D: Local (Ollama)
        elif self.provider == "Local (Ollama)":
            try:
                url = self.base_url if self.base_url else "http://localhost:11434"
                llm = Ollama(model=self.model_name, base_url=url)
                return llm.invoke(prompt)
            except Exception as e:
                return f"Local LLM Error: Ensure Ollama is running. Details: {str(e)}"
        
        return "Invalid Provider Configuration."

    def determine_test_file_path(self, original_path: str, target_lang: str) -> str:
        directory, filename = os.path.split(original_path)
        name_part, ext = os.path.splitext(filename)
        new_filename = f"{name_part}Tests{ext}" if target_lang == "C#" else f"{name_part}.test{ext}"
        return os.path.join(directory, new_filename)

    def sanitize_code_block(self, raw_llm_output: str) -> str:
        if "```" in raw_llm_output:
            lines = raw_llm_output.splitlines()
            code_lines = []
            inside_block = False
            for line in lines:
                if line.startswith("```"):
                    inside_block = not inside_block
                    continue
                if inside_block:
                    code_lines.append(line)
            return "\n".join(code_lines)
        return raw_llm_output

# --- WEB UI INTERFACE ---
st.set_page_config(page_title="Multi-LLM Test Agent", layout="wide")
st.title("🛠️ Enterprise Test Case Generator Agent")

# Sidebar - Dynamic Settings Matrix
st.sidebar.header("🔌 LLM Settings")
provider = st.sidebar.selectbox(
    "LLM Provider", 
    ["Local (Ollama)", "Cloud (OpenAI)", "DeepSeek", "Anthropic (Claude)"]
)

# Contextual form layouts matching SDK rules
if provider == "Anthropic (Claude)":
    model_name = st.sidebar.selectbox("Model", ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"])
    api_key = st.sidebar.text_input("Anthropic API Key", type="password")
    base_url = None
elif provider == "Cloud (OpenAI)":
    model_name = st.sidebar.selectbox("Model", ["gpt-4o", "gpt-4-turbo"])
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    base_url = None
elif provider == "DeepSeek":
    model_name = st.sidebar.selectbox("Model", ["deepseek-coder", "deepseek-chat"])
    api_key = st.sidebar.text_input("DeepSeek API Key", type="password")
    base_url = None
else:
    model_name = st.sidebar.text_input("Ollama Model Name", value="qwen2.5-coder")
    base_url = st.sidebar.text_input("Ollama Endpoint", value="http://localhost:11434")
    api_key = None

st.sidebar.header("🎯 Language Target")
target_lang = st.sidebar.radio("Primary Language Syntax", ["C#", "TypeScript"])

agent = TestGeneratorAgent(provider, model_name, api_key, base_url)

tab1, tab2 = st.tabs(["📂 Option A: Scan & Write via Local File Path", "📝 Option B: Web UI Copy-Paste"])

# --- OPTION A: SCAN & WRITE FILE ---
with tab1:
    st.header("Scan File and Save to Project Tree")
    file_path = st.text_input("Absolute File Path:", placeholder="e.g., /Users/dev/Project/Services/OrderService.cs")
    
    if st.button("Generate & Write Test File"):
        if not file_path or not os.path.exists(file_path):
            st.error("Error: Please check the file path. File does not exist.")
        else:
            with st.spinner(f"Agent compiling architecture via {provider}..."):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        source_code = f.read()
                    
                    raw_output = agent.generate_tests(target_lang, source_code)
                    clean_test_code = agent.sanitize_code_block(raw_output)
                    
                    destination_path = agent.determine_test_file_path(file_path, target_lang)
                    with open(destination_path, "w", encoding="utf-8") as test_file:
                        test_file.write(clean_test_code)
                        
                    st.success(f"🎉 Success! Generated test file saved to disk.")
                    st.info(f"Target location: `{destination_path}`")
                    st.code(clean_test_code, language=target_lang.lower())
                except Exception as e:
                    st.error(f"Execution failed: {str(e)}")

# --- OPTION B: QUICK WEB COPY-PASTE ---
with tab2:
    st.header("Web Code Snippet Sandbox")
    source_code_input = st.text_area("Paste code methods/classes here:", height=250)
    
    if st.button("Generate Tests on Screen"):
        if not source_code_input.strip():
            st.error("Please insert code snippets before starting.")
        else:
            with st.spinner(f"Processing generation rules via {provider}..."):
                raw_output = agent.generate_tests(target_lang, source_code_input)
                clean_test_code = agent.sanitize_code_block(raw_output)
                st.success("Generation complete!")
                st.code(clean_test_code, language=target_lang.lower())