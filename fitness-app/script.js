// Optimized Scroll Effect with Class Toggle
const handleScroll = () => {
    const navContainer = document.querySelector('.nav-container');
    const scrollY = window.scrollY || window.pageYOffset;
    
    requestAnimationFrame(() => {
        const scrolled = scrollY > 60;
        
        if(navContainer) {
            const progress = Math.min(scrollY / 100, 1);
            navContainer.style.opacity = scrolled ? progress : 0;
            navContainer.style.transform = `scale(${scrolled ? 1 : 0.98})`;
            navContainer.style.background = scrolled 
                ? 'rgba(255,255,255,0.1)'
                : 'transparent';
            navContainer.style.backdropFilter = scrolled 
                ? 'blur(20px)'
                : 'none';
        }
    });
};

// Throttled Scroll Event Listener
let isScrolling;
window.addEventListener('scroll', () => {
    window.cancelAnimationFrame(isScrolling);
    isScrolling = window.requestAnimationFrame(handleScroll);
});

// Enhanced Button Handling
document.addEventListener('DOMContentLoaded', () => {
    // Navigation Buttons
    const setupButton = (selector, path) => {
        try {
            document.querySelectorAll(selector).forEach(btn => {
                btn.addEventListener('click', () => {
                    window.location.href = path;
                });
            });
        } catch (error) {
            console.error(`Button error (${selector}):`, error);
        }
    };

    setupButton('.workout-btn', '/workout-planner');
    setupButton('.nutrition-btn', '/nutrition-plan');

    // Mobile Menu Functions
    window.openMenu = () => {
        document.getElementById('sideMenu').classList.remove('translate-x-full');
    };
    
    window.closeMenu = () => {
        document.getElementById('sideMenu').classList.add('translate-x-full');
    };
});

// Optional: Close menu on outside click
document.addEventListener('click', (e) => {
    const sideMenu = document.getElementById('sideMenu');
    const menuBtn = document.querySelector('.mobile-menu-btn');
    
    if(!sideMenu.contains(e.target) && !menuBtn.contains(e.target)) {
        sideMenu.classList.add('translate-x-full');
    }
});