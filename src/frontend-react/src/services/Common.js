//export const BASE_API_URL = process.env.REACT_APP_BASE_API_URL;
export const BASE_API_URL = process.env.NEXT_PUBLIC_BASE_API_URL;
export const APP_VERSION = 1.0;
export const MOCK_SERVICE = true;

export function epochToJsDate(ts) {
    let dt = new Date(ts)
    return dt.toLocaleDateString() + " " + dt.toLocaleTimeString();
}

export function formatRelativeTime(epochTimestamp) {
    const now = Date.now();
    // Convert epoch seconds to milliseconds by multiplying by 1000
    const timestamp = epochTimestamp * 1000;
    const diff = now - timestamp;

    // Convert time differences to minutes, hours, days
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    // Format relative time based on difference
    if (minutes < 1) {
        return 'just now';
    } else if (minutes < 60) {
        return `${minutes} ${minutes === 1 ? 'minute' : 'minutes'} ago`;
    } else if (hours < 24) {
        return `${hours} ${hours === 1 ? 'hour' : 'hours'} ago`;
    } else if (days < 7) {
        return `${days} ${days === 1 ? 'day' : 'days'} ago`;
    } else if (days < 30) {
        const weeks = Math.floor(days / 7);
        return `${weeks} ${weeks === 1 ? 'week' : 'weeks'} ago`;
    } else {
        // For older dates, show month and year
        return new Date(timestamp).toLocaleDateString('en-US', {
            month: 'long',
            year: 'numeric'
        });
    }
};

export function uuid() {
    const newUuid = ([1e7] + -1e3 + -4e3 + -8e3 + -1e11).replace(/[018]/g, (c) =>
        (c ^ (crypto.getRandomValues(new Uint8Array(1))[0] & (15 >> (c / 4)))).toString(16),
    )
    return newUuid;
}