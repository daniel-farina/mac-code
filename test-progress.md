# Mac Code Agent - Test Progress

## Test 1: Calculator App (demo01/)
- **Status**: Working
- **Result**: Agent scaffolded vite + installed deps + created 2 files in ONE turn (3,023 tokens, 20 tok/s)
- **Improvement**: Agentic system prompt - "complete ALL steps in one response"
- **Before**: 3 turns needed. **After**: 1 turn, 4 chained operations

## Test 2: Self-improvement
- **Status**: Working
- **Result**: Agent read index.html + main.js, created style.css (192 lines glassmorphism), added 35 lines to main.js
- **Behavior**: Correctly identified missing CSS and added keyboard support without being told what was wrong

## Test 3: WebMCP Gmail
- **Status**: Working (when extension connected)
- **Result**: Agent navigated to Gmail, read page, listed 5 emails with senders + subjects in 3 MCP calls
- **Search test**: MCP timed out on second session (extension disconnect), agent tried opening Chrome as fallback
- **36 iterations** on first test - agent kept using MCP tools autonomously

## Test 4: Edge cases (in progress)
- Multi-file project edits: working (4 files in demo01/)
- Error recovery: agent opened Chrome when MCP failed
- Background jobs: vite dev server auto-detected and backgrounded

## Agent Improvements Made
1. Agentic system prompt - autonomous task completion in one response
2. Quiet mode with tok/s display
3. Claude Code-style panels (Update, Create, Bash)
4. Unified diff (DIFF:) as default edit format
5. Clean prompt (just > instead of auto 9b >)
6. Background job URL detection
7. Session persistence (auto-save/resume)
8. npm create not backgrounded, 300s timeout
9. Status indicators (coding... with tok/s)

## Total commits: 30+
