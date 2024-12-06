import { BASE_API_URL, uuid } from "./Common";
import axios from 'axios';

// Create an axios instance with base configuration
const api = axios.create({
    baseURL: BASE_API_URL
});
// Add request interceptor to include session ID in headers
api.interceptors.request.use((config) => {
    const sessionId = localStorage.getItem('userSessionId');
    if (sessionId) {
        config.headers['X-Session-ID'] = sessionId;
    }
    return config;
}, (error) => {
    return Promise.reject(error);
});

const DataService = {
    Init: function () {
        // Any application initialization logic comes here
    },
    GetPodcasts: async function (limit) {
        return await api.get(BASE_API_URL + "/podcasts?limit=" + limit);
    },
    GetPodcast: async function (podcast_id) {
        return await api.get(BASE_API_URL + "/podcasts/" + podcast_id);
    },
    GetPodcastAudio: function (audio_path) {
        return BASE_API_URL + "/podcasts/audio/" + audio_path;
    },
    GetNewsletters: async function (limit) {
        return await api.get(BASE_API_URL + "/newsletters?limit=" + limit);
    },
    GetNewsletter: async function (newsletter_id) {
        return await api.get(BASE_API_URL + "/newsletters/" + newsletter_id);
    },
    GetNewsletterImage: function (image_path) {
        return BASE_API_URL + "/newsletters/image/" + image_path;
    },
    GetChats: async function (model, limit) {
        return await api.get(BASE_API_URL + "/" + model + "/chats?limit=" + limit);
    },
    GetChat: async function (model, chat_id) {
        return await api.get(BASE_API_URL + "/" + model + "/chats/" + chat_id);
    },
    StartChatWithLLM: async function (model, message) {
        return await api.post(BASE_API_URL + "/" + model + "/chats/", message);
    },
    ContinueChatWithLLM: async function (model, chat_id, message) {
        return await api.post(BASE_API_URL + "/" + model + "/chats/" + chat_id, message);
    },
    GetChatMessageImage: function (model, image_path) {
        return BASE_API_URL + "/" + model + "/" + image_path;
    },
}

export default DataService;