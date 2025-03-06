document.addEventListener("DOMContentLoaded", () => {
  const featureCards = document.querySelectorAll(".feature-card");
  const listItems = document.querySelectorAll(".how-list li");

  const revealOnScroll = (elements) => {
    const triggerBottom = window.innerHeight * 0.85;
    elements.forEach((el) => {
      const elementTop = el.getBoundingClientRect().top;
      if (elementTop < triggerBottom) {
        el.classList.add("visible");
      } else {
        el.classList.remove("visible");
      }
    });
  };

  window.addEventListener("scroll", () => {
    revealOnScroll(featureCards);
    revealOnScroll(listItems);
  });

  revealOnScroll(featureCards);
  revealOnScroll(listItems);
});
