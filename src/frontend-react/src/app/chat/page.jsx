// 'use client';

// import { useState, use, useEffect } from 'react';
// import { useRouter } from 'next/navigation';
// import IconButton from '@mui/material/IconButton';
// import ChatInput from '@/components/chat/ChatInput';
// import ChatHistory from '@/components/chat/ChatHistory';
// import ChatHistorySidebar from '@/components/chat/ChatHistorySidebar';
// import ChatMessage from '@/components/chat/ChatMessage';
// //import DataService from "../../services/MockDataService"; // Mock
// import DataService from "../../services/DataService";
// import { uuid } from "../../services/Common";

// // Import the styles
// import styles from "./styles.module.css";

// export default function ChatPage({ searchParams }) {
//     const params = use(searchParams);
//     const chat_id = params.id;
//     const model = params.model || 'llm';
//     console.log(chat_id, model);

//     // Component States
//     const [chatId, setChatId] = useState(params.id);
//     const [hasActiveChat, setHasActiveChat] = useState(false);
//     const [chat, setChat] = useState(null);
//     const [refreshKey, setRefreshKey] = useState(0);
//     const [isTyping, setIsTyping] = useState(false);
//     const [selectedModel, setSelectedModel] = useState(model);
//     const router = useRouter();

//     const fetchChat = async (id) => {
//         try {
//             setChat(null);
//             const response = await DataService.GetChat(model, id);
//             setChat(response.data);
//             console.log(chat);
//         } catch (error) {
//             console.error('Error fetching chat:', error);
//             setChat(null);
//         }
//     };

//     // Setup Component
//     useEffect(() => {
//         if (chat_id) {
//             fetchChat(chat_id);
//             setHasActiveChat(true);
//         } else {
//             setChat(null);
//             setHasActiveChat(false);
//         }
//     }, [chat_id]);
//     useEffect(() => {
//         setSelectedModel(model);
//     }, [model]);

//     function tempChatMessage(message) {
//         // Set temp values
//         message["message_id"] = uuid();
//         message["role"] = 'user';
//         if (chat) {
//             // Append message
//             var temp_chat = { ...chat };
//             temp_chat["messages"].push(message);
//         } else {
//             var temp_chat = {
//                 "messages": [message]
//             }
//             return temp_chat;
//         }
//     }

//     // Handlers
//     const newChat = (message) => {
//         console.log(message);
//         // Start a new chat and submit to LLM
//         const startChat = async (message) => {
//             try {
//                 // Show typing indicator
//                 setIsTyping(true);
//                 setHasActiveChat(true);
//                 setChat(tempChatMessage(message)); // Show the user input message while LLM is invoked

//                 // Submit chat
//                 const response = await DataService.StartChatWithLLM(model, message);
//                 console.log(response.data);

//                 // Hide typing indicator and add response
//                 setIsTyping(false);

//                 setChat(response.data);
//                 setChatId(response.data["chat_id"]);
//                 router.push('/chat?model=' + selectedModel + '&id=' + response.data["chat_id"]);
//             } catch (error) {
//                 console.error('Error fetching chat:', error);
//                 setIsTyping(false);
//                 setChat(null);
//                 setChatId(null);
//                 setHasActiveChat(false);
//                 router.push('/chat?model=' + selectedModel)
//             }
//         };
//         startChat(message);

//     };
//     const appendChat = (message) => {
//         console.log(message);
//         // Append message and submit to LLM

//         const continueChat = async (id, message) => {
//             try {
//                 // Show typing indicator
//                 setIsTyping(true);
//                 setHasActiveChat(true);
//                 tempChatMessage(message);

//                 // Submit chat
//                 const response = await DataService.ContinueChatWithLLM(model, id, message);
//                 console.log(response.data);

//                 // Hide typing indicator and add response
//                 setIsTyping(false);

//                 setChat(response.data);
//                 forceRefresh();
//             } catch (error) {
//                 console.error('Error fetching chat:', error);
//                 setIsTyping(false);
//                 setChat(null);
//                 setHasActiveChat(false);
//             }
//         };
//         continueChat(chat_id, message);
//     };
//     // Force re-render by updating the key
//     const forceRefresh = () => {
//         setRefreshKey(prevKey => prevKey + 1);
//     };
//     const handleModelChange = (newValue) => {

//         setSelectedModel(newValue);
//         var path = '/chat?model=' + newValue;
//         if (chat_id) {
//             path = path + '&id=' + chat_id;
//         }
//         router.push(path)
//     };

//     return (
//         <div className={styles.container}>

//             {/* Hero Section */}
//             {!hasActiveChat && (
//                 <section className={styles.hero}>
//                     <div className={styles.heroContent}>
//                         <h1>Pattern Assistant 🌟</h1>
//                         {/* Main Chat Input: ChatInput */}
//                         <ChatInput onSendMessage={newChat} className={styles.heroChatInputContainer} selectedModel={selectedModel} onModelChange={handleModelChange}></ChatInput>
//                     </div>
//                 </section>
//             )}

//             {/* Chat History Section: ChatHistory */}
//             {!hasActiveChat && (
//                 <ChatHistory model={model}></ChatHistory>
//             )}

//             {/* Chat Block Header Section */}
//             {hasActiveChat && (
//                 <div className={styles.chatHeader}></div>
//             )}
//             {/* Active Chat Interface */}
//             {hasActiveChat && (
//                 <div className={styles.chatInterface}>
//                     {/* Chat History Sidebar: ChatHistorySidebar */}
//                     <ChatHistorySidebar chat_id={chat_id} model={model}></ChatHistorySidebar>

//                     {/* Main chat area */}
//                     <div className={styles.mainContent}>
//                         {/* Chat message: ChatMessage */}
//                         <ChatMessage chat={chat} key={refreshKey} isTyping={isTyping} model={model}></ChatMessage>
//                         {/* Sticky chat input area: ChatInput */}
//                         <ChatInput
//                             onSendMessage={appendChat}
//                             chat={chat}
//                             selectedModel={selectedModel}
//                             onModelChange={setSelectedModel}
//                             disableModelSelect={true}
//                         ></ChatInput>
//                     </div>
//                 </div>
//             )}
//         </div>
//     );
// }

// 'use client';

// import { useState, useEffect } from 'react';
// import { useRouter } from 'next/navigation';
// import ChatInput from '@/components/chat/ChatInput';
// import ChatHistory from '@/components/chat/ChatHistory';
// import ChatHistorySidebar from '@/components/chat/ChatHistorySidebar';
// import ChatMessage from '@/components/chat/ChatMessage';
// import DataService from '../../services/DataService';
// import { uuid } from '../../services/Common';

// // Styles
// import styles from './styles.module.css';

// export default function ChatPage({ searchParams }) {
//     const params = searchParams;
//     const router = useRouter();
//     const model = params.model || 'llm';
//     const chat_id = params.id;

//     const [chat, setChat] = useState(null);
//     const [selectedModel, setSelectedModel] = useState(model);
//     const [hasActiveChat, setHasActiveChat] = useState(!!chat_id);

//     const fetchChat = async (id) => {
//         try {
//             const response = await DataService.GetChat(model, id);
//             setChat(response.data);
//         } catch (error) {
//             console.error('Error fetching chat:', error);
//         }
//     };

//     useEffect(() => {
//         if (chat_id) fetchChat(chat_id);
//     }, [chat_id]);

//     const startNewChat = async (message) => {
//         try {
//             const response = await DataService.StartChatWithLLM(model, message);
//             const newChatId = response.data.chat_id;
//             setChat(response.data);
//             setHasActiveChat(true);
//             router.push(`/chat?model=${model}&id=${newChatId}`);
//         } catch (error) {
//             console.error('Error starting new chat:', error);
//         }
//     };

//     const appendChat = async (message) => {
//         try {
//             const response = await DataService.ContinueChatWithLLM(model, chat_id, message);
//             setChat(response.data);
//         } catch (error) {
//             console.error('Error appending chat:', error);
//         }
//     };

//     return (
//         <div className={styles.container}>
//             {!hasActiveChat && (
//                 <section className={styles.hero}>
//                     <div className={styles.heroContent}>
//                         <h1>Pattern Assistant 🌟</h1>
//                         <ChatInput
//                             onSendMessage={startNewChat}
//                             selectedModel={selectedModel}
//                             onModelChange={(newModel) => setSelectedModel(newModel)}
//                         />
//                     </div>
//                 </section>
//             )}
//             {hasActiveChat && (
//                 <div className={styles.chatInterface}>
//                     <ChatHistorySidebar chat_id={chat_id} model={model} />
//                     <div className={styles.mainContent}>
//                         <ChatMessage chat={chat} model={model} />
//                         <ChatInput
//                             onSendMessage={appendChat}
//                             selectedModel={selectedModel}
//                             disableModelSelect={true} // Enables "New Pattern" button
//                         />
//                     </div>
//                 </div>
//             )}
//         </div>
//     );
// }

// 'use client';

// import { useState, useEffect } from 'react';
// import { useRouter } from 'next/navigation';
// import ChatInput from '@/components/chat/ChatInput';
// import ChatHistorySidebar from '@/components/chat/ChatHistorySidebar';
// import ChatMessage from '@/components/chat/ChatMessage';
// import DataService from '../../services/DataService';
// import { uuid } from '../../services/Common';

// // Styles
// import styles from './styles.module.css';

// export default function ChatPage({ searchParams }) {
//     const params = searchParams;
//     const router = useRouter();
//     const model = params.model || 'llm';
//     const chat_id = params.id;

//     const [chat, setChat] = useState(null);
//     const [selectedModel, setSelectedModel] = useState(model);
//     const [hasActiveChat, setHasActiveChat] = useState(!!chat_id);
//     const [isTyping, setIsTyping] = useState(false);

//     const fetchChat = async (id) => {
//         try {
//             const response = await DataService.GetChat(model, id);
//             setChat(response.data);
//         } catch (error) {
//             console.error('Error fetching chat:', error);
//         }
//     };

//     useEffect(() => {
//         if (chat_id) fetchChat(chat_id);
//     }, [chat_id]);

//     const startNewChat = async (message) => {
//         // Generate a temporary chat ID and redirect immediately
//         const tempChatId = uuid();
//         const tempChat = {
//             chat_id: tempChatId,
//             messages: [{ ...message, role: 'user', message_id: uuid() }],
//         };

//         setChat(tempChat); // Show temporary chat
//         setHasActiveChat(true);
//         router.push(`/chat?model=${model}&id=${tempChatId}`); // Redirect immediately

//         try {
//             // Process the actual chat backend logic
//             const response = await DataService.StartChatWithLLM(model, message);
//             const newChatId = response.data.chat_id;
//             setChat(response.data); // Replace temporary chat with real data
//             router.push(`/chat?model=${model}&id=${newChatId}`); // Ensure proper URL
//         } catch (error) {
//             console.error('Error starting new chat:', error);
//             setIsTyping(false);
//         }
//     };

//     const appendChat = async (message) => {
//         try {
//             setIsTyping(true);
//             const tempChat = { ...chat };
//             tempChat.messages.push({ ...message, role: 'user', message_id: uuid() });
//             setChat(tempChat); // Update chat UI immediately

//             const response = await DataService.ContinueChatWithLLM(model, chat_id, message);
//             setChat(response.data); // Replace with backend response
//         } catch (error) {
//             console.error('Error appending chat:', error);
//         } finally {
//             setIsTyping(false);
//         }
//     };

//     return (
//         <div className={styles.container}>
//             {!hasActiveChat && (
//                 <section className={styles.hero}>
//                     <div className={styles.heroContent}>
//                         <h1>Pattern Assistant 🌟</h1>
//                         <ChatInput
//                             onSendMessage={startNewChat}
//                             selectedModel={selectedModel}
//                             onModelChange={(newModel) => setSelectedModel(newModel)}
//                         />
//                     </div>
//                 </section>
//             )}
//             {hasActiveChat && (
//                 <div className={styles.chatInterface}>
//                     <ChatHistorySidebar chat_id={chat_id} model={model} />
//                     <div className={styles.mainContent}>
//                         <ChatMessage chat={chat} model={model} isTyping={isTyping} />
//                         <ChatInput
//                             onSendMessage={appendChat}
//                             selectedModel={selectedModel}
//                             disableModelSelect={true}
//                         />
//                     </div>
//                 </div>
//             )}
//         </div>
//     );
// }

'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import ChatHistorySidebar from '@/components/chat/ChatHistorySidebar';
import ChatMessage from '@/components/chat/ChatMessage';
import ChatInput from '@/components/chat/ChatInput';
import DataService from '../../services/DataService';
import { uuid } from '../../services/Common';

// Styles
import styles from './styles.module.css';

export default function ChatPage({ searchParams }) {
    const params = searchParams;
    const router = useRouter();
    const model = params.model || 'llm';
    const chat_id = params.id;

    const [chat, setChat] = useState(null); // Current chat data
    const [hasActiveChat, setHasActiveChat] = useState(!!chat_id); // Check if there's an active chat

    // Fetch chat details when there's a chat ID
    const fetchChat = async (id) => {
        try {
            setChat(null); // Clear the current chat state
            const response = await DataService.GetChat(model, id);
            setChat(response.data); // Set the fetched chat data
        } catch (error) {
            console.error('Error fetching chat:', error);
            setChat(null); // Reset in case of an error
        }
    };

    useEffect(() => {
        if (chat_id) {
            fetchChat(chat_id); // Fetch chat when chat_id exists
            setHasActiveChat(true);
        } else {
            setHasActiveChat(false); // No active chat
        }
    }, [chat_id]);

    // Start a new chat and redirect to the chat page
    const startNewChat = async (message) => {
        const tempChatId = uuid(); // Generate a temporary chat ID
        const tempChat = {
            chat_id: tempChatId,
            messages: [{ ...message, role: 'user', message_id: uuid() }],
        };

        // Redirect to chat page with a temporary chat state
        setChat(tempChat);
        setHasActiveChat(true);
        router.push(`/chat?model=${model}&id=${tempChatId}`); // Navigate to chat page

        try {
            // Make the backend call to start the chat
            const response = await DataService.StartChatWithLLM(model, message);
            const newChatId = response.data.chat_id;

            // Replace the temporary chat with actual data
            setChat(response.data);
            router.push(`/chat?model=${model}&id=${newChatId}`);
        } catch (error) {
            console.error('Error starting new chat:', error);
        }
    };

    return (
        <div className={styles.container}>
            {!hasActiveChat ? (
                /* Input Page */
                <section className={styles.hero}>
                    <div className={styles.heroContent}>
                        <h1>Pattern Assistant 🌟</h1>
                        <ChatInput
                            onSendMessage={startNewChat} // Use startNewChat to handle submissions
                            selectedModel={model}
                            disableModelSelect={false}
                        />
                    </div>
                </section>
            ) : (
                /* Chat Page */
                <div className={styles.chatInterface}>
                    <ChatHistorySidebar chat_id={chat_id} model={model} />
                    <div className={styles.mainContent}>
                        <ChatMessage chat={chat} model={model} />
                        <ChatInput disableModelSelect={true} /> {/* Only show "New Pattern" button */}
                    </div>
                </div>
            )}
        </div>
    );
}
