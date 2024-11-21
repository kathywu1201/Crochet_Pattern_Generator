import { uuid } from "./Common";
import { newsletters, episodes, recentChats } from "./SampleData";

const DataService = {
    Init: function () {
        console.log('Mock Data Service Initialized');
        return Promise.resolve();
    },

    GetPodcasts: async function (limit) {
        const limitedEpisodes = limit ? episodes.slice(0, limit) : episodes;
        return Promise.resolve({ data: limitedEpisodes });
    },

    GetNewsletters: async function (limit) {
        const limitedNewsletters = limit ? newsletters.slice(0, limit) : newsletters;
        return Promise.resolve({ data: limitedNewsletters });
    },

    GetChats: async function (limit) {
        // const limitedChats = limit ? recentChats.slice(0, limit) : recentChats;
        // return Promise.resolve({ data: limitedChats });

        // Create a copy of recentChats to avoid modifying the original array
        const sortedChats = [...recentChats].sort((a, b) => {
            // Sort in descending order (most recent first)
            return b.dts - a.dts;
        }
        );

        const limitedChats = limit ? sortedChats.slice(0, limit) : sortedChats;
        return Promise.resolve({ data: limitedChats });
    },

    GetChat: async function (id) {
        var chat_response = recentChats.find(chat => chat.id === id);
        return Promise.resolve({ data: chat_response });
    },
    StartChatWithLLM: async function (message) {
        // Simulate a delay to mimic API response time
        await new Promise(resolve => setTimeout(resolve, 200));

        // Generate unique id
        var id = uuid();
        var now = Date.now();
        now = parseInt(now / 1000);

        message["id"] = uuid()

        // Mock response
        var chat_response = {
            "id": id,
            "title": message["content"],
            "dts": now,
            "messages": [
                message,
                {
                    "id": uuid(),
                    role: 'assistant',
                    content: 'This is a response to your message: ' + message["content"]
                }
            ]
        }

        // add to recent chats
        recentChats.push(chat_response);

        console.log(chat_response);
        return Promise.resolve({ data: chat_response });
    },
    ChatWithLLM: async function (id, message) {
        // Simulate a delay to mimic API response time
        await new Promise(resolve => setTimeout(resolve, 200));

        // Get chat
        var chat_response = recentChats.find(chat => chat.id === id);
        var now = Date.now();
        now = parseInt(now / 1000);
        chat_response["dts"] = now;

        message["id"] = uuid()

        chat_response["messages"].push(message);

        chat_response["messages"].push({
            "id": uuid(),
            role: 'assistant',
            content: 'This is a response to your message: ' + message["content"]
        })

        return Promise.resolve({ data: chat_response });
    }
};

export default DataService;