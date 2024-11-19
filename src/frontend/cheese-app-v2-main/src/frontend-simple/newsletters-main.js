// DataService configuration
const DataService = {
    baseURL: 'http://localhost:9000',

    async GetNewsletters(limit = 100) {
        try {
            const response = await axios.get(`${this.baseURL}/newsletters?limit=${limit}`);
            return response;
        } catch (error) {
            console.error('Error fetching newsletters:', error);
            throw error;
        }
    },

    GetNewsletterImage(imagePath) {
        return `${this.baseURL}/newsletters/image/${imagePath}`;
    }
};

// DOM Elements
const newsletterGrid = document.getElementById('newsletter-grid');

// Create newsletter card
function createNewsletterCard(newsletter) {
    const card = document.createElement('article');
    card.className = 'card';
    card.onclick = () => handleNewsletterClick(newsletter.id);

    card.innerHTML = `
        <div class="image-container">
            <img
                src="${DataService.GetNewsletterImage(newsletter.image)}"
                alt="${newsletter.title}"
                class="image"
            />
            <span class="category">${newsletter.category}</span>
        </div>
        <div class="content">
            <div class="meta">
                <span class="date">${newsletter.date}</span>
                <span class="read-time">${newsletter.readTime}</span>
            </div>
            <h3 class="title">${newsletter.title}</h3>
            <p class="excerpt">${newsletter.excerpt}</p>
        </div>
    `;

    return card;
}

// Initialize the page
async function initializePage() {
    try {

        // Fetch all newsletters
        const response = await DataService.GetNewsletters();
        const newsletters = response.data;

        // Populate newsletter grid
        newsletters.forEach(newsletter => {
            const card = createNewsletterCard(newsletter);
            newsletterGrid.appendChild(card);
        });

    } catch (error) {
        console.error('Error initializing app:', error);
    }
}

initializePage();
