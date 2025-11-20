import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeSanitize from 'rehype-sanitize'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [showSources, setShowSources] = useState({})
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleStreamingChat = async () => {
    if (!input.trim()) return

    const userMessage = { role: 'user', content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsStreaming(true)

    // 添加一個空的助手訊息，用於接收串流內容
    const assistantMessageIndex = messages.length + 1
    const userInput = input
    setMessages(prev => [...prev, { role: 'assistant', content: '', sources: null }])

    try {
      const response = await fetch(
        `http://localhost:8000/api/chat/stream?message=${encodeURIComponent(userInput)}`
      )

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              if (data.done) {
                setIsStreaming(false)
                // 串流完成後，獲取來源資訊
                try {
                  const sourcesResponse = await fetch('http://localhost:8000/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: userInput })
                  })
                  const sourcesData = await sourcesResponse.json()
                  setMessages(prev => {
                    const newMessages = [...prev]
                    newMessages[assistantMessageIndex] = {
                      ...newMessages[assistantMessageIndex],
                      sources: sourcesData.sources || []
                    }
                    return newMessages
                  })
                } catch (e) {
                  console.error('Error fetching sources:', e)
                }
              } else {
                setMessages(prev => {
                  const newMessages = [...prev]
                  newMessages[assistantMessageIndex] = {
                    role: 'assistant',
                    content: newMessages[assistantMessageIndex].content + data.content,
                    sources: newMessages[assistantMessageIndex].sources
                  }
                  return newMessages
                })
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e)
            }
          }
        }
      }
    } catch (error) {
      console.error('Streaming error:', error)
      setIsStreaming(false)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '抱歉，發生錯誤，請稍後再試。'
      }])
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    handleStreamingChat()
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto max-w-4xl h-screen flex flex-col p-4">
        {/* Header */}
        <div className="bg-white rounded-t-lg shadow-md p-6 mb-0">
          <h1 className="text-3xl font-bold text-indigo-600">
            TTU CSE 聊天機器人
          </h1>
          <p className="text-gray-600 mt-2">
            大同大學資訊工程學系智能助手
          </p>
        </div>

        {/* Messages Container */}
        <div className="flex-1 bg-white shadow-md overflow-y-auto p-6 space-y-4">
          {messages.length === 0 ? (
            <div className="text-center text-gray-500 mt-20">
              <div className="text-6xl mb-4">💬</div>
              <p className="text-xl">開始對話吧！</p>
              <p className="text-sm mt-2">問我關於大同大學資工系的任何問題</p>
            </div>
          ) : (
            messages.map((message, index) => (
              <div
                key={index}
                className={`flex ${
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                <div
                  className={`max-w-[70%] rounded-lg p-4 ${
                    message.role === 'user'
                      ? 'bg-indigo-600 text-white'
                      : 'bg-gray-100 text-gray-800'
                  }`}
                >
                  <div className="flex items-start space-x-2">
                    <div className="text-sm font-semibold mb-2">
                      {message.role === 'user' ? '你' : 'AI 助手'}
                    </div>
                  </div>
                  {message.role === 'user' ? (
                    <div className="whitespace-pre-wrap">{message.content}</div>
                  ) : (
                    <>
                      <div className="prose prose-sm max-w-none prose-invert:text-gray-800">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          rehypePlugins={[rehypeSanitize]}
                          components={{
                            // 自訂樣式
                            p: ({node, ...props}) => <p className="mb-2 last:mb-0" {...props} />,
                            ul: ({node, ...props}) => <ul className="list-disc ml-4 mb-2" {...props} />,
                            ol: ({node, ...props}) => <ol className="list-decimal ml-4 mb-2" {...props} />,
                            li: ({node, ...props}) => <li className="mb-1" {...props} />,
                            code: ({node, inline, ...props}) => 
                              inline 
                                ? <code className="bg-gray-200 px-1 py-0.5 rounded text-sm" {...props} />
                                : <code className="block bg-gray-200 p-2 rounded my-2 overflow-x-auto" {...props} />,
                            pre: ({node, ...props}) => <pre className="bg-gray-200 p-2 rounded my-2 overflow-x-auto" {...props} />,
                            a: ({node, ...props}) => <a className="text-blue-600 hover:underline" target="_blank" rel="noopener noreferrer" {...props} />,
                            h1: ({node, ...props}) => <h1 className="text-xl font-bold mb-2 mt-3" {...props} />,
                            h2: ({node, ...props}) => <h2 className="text-lg font-bold mb-2 mt-2" {...props} />,
                            h3: ({node, ...props}) => <h3 className="text-base font-bold mb-1 mt-2" {...props} />,
                            blockquote: ({node, ...props}) => <blockquote className="border-l-4 border-gray-400 pl-3 italic my-2" {...props} />,
                            table: ({node, ...props}) => <table className="border-collapse border border-gray-300 my-2" {...props} />,
                            th: ({node, ...props}) => <th className="border border-gray-300 px-2 py-1 bg-gray-100 font-semibold" {...props} />,
                            td: ({node, ...props}) => <td className="border border-gray-300 px-2 py-1" {...props} />,
                          }}
                        >
                          {message.content}
                        </ReactMarkdown>
                      </div>
                      {message.sources && message.sources.length > 0 && (
                        <div className="mt-3 pt-3 border-t border-gray-300">
                          <button
                            onClick={() => setShowSources(prev => ({
                              ...prev,
                              [index]: !prev[index]
                            }))}
                            className="flex items-center text-xs text-gray-600 hover:text-gray-800 font-medium transition-colors"
                          >
                            <svg 
                              className={`w-4 h-4 mr-1 transition-transform ${showSources[index] ? 'rotate-90' : ''}`}
                              fill="none" 
                              stroke="currentColor" 
                              viewBox="0 0 24 24"
                            >
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                            </svg>
                            {showSources[index] ? '隱藏' : '查看'}資料來源 ({message.sources.length})
                          </button>
                          {showSources[index] && (
                            <div className="mt-2 space-y-2">
                              {message.sources.map((source, idx) => (
                                <div key={idx} className="bg-gray-50 p-3 rounded text-xs">
                                  <div className="flex justify-between items-start mb-1">
                                    <span className="font-semibold text-gray-700">來源 {idx + 1}</span>
                                    <div className="flex gap-2">
                                      {source.relevance !== undefined && (
                                        <span className="bg-blue-100 text-blue-700 px-2 py-0.5 rounded">
                                          相關度: {(source.relevance * 100).toFixed(1)}%
                                        </span>
                                      )}
                                      {source.rerank_score !== undefined && (
                                        <span className="bg-green-100 text-green-700 px-2 py-0.5 rounded">
                                          重排分數: {source.rerank_score.toFixed(3)}
                                        </span>
                                      )}
                                    </div>
                                  </div>
                                  <div className="text-gray-600 mb-1">
                                    <span className="font-medium">文件:</span> {source.source}
                                  </div>
                                  <div className="text-gray-700 bg-white p-2 rounded border border-gray-200">
                                    {source.content}
                                  </div>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            ))
          )}
          {isStreaming && (
            <div className="flex justify-start">
              <div className="bg-gray-100 rounded-lg p-4">
                <div className="flex space-x-2">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200"></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Form */}
        <form onSubmit={handleSubmit} className="bg-white rounded-b-lg shadow-md p-4">
          <div className="flex space-x-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="輸入您的問題..."
              disabled={isStreaming}
              className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:bg-gray-100"
            />
            <button
              type="submit"
              disabled={isStreaming || !input.trim()}
              className="px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-semibold"
            >
              {isStreaming ? '發送中...' : '發送'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default App
