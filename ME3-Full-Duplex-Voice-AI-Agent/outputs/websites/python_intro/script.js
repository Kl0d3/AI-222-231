document.addEventListener('DOMContentLoaded', () => {
    const runBtn = document.getElementById('run-btn');
    const codeInput = document.getElementById('code-input');
    const consoleOutput = document.getElementById('console-output');

    if (runBtn && codeInput && consoleOutput) {
        runBtn.addEventListener('click', () => {
            consoleOutput.textContent = "Executing code simulation...\n\n";
            
            setTimeout(() => {
                const code = codeInput.value;
                if (code.includes("factorial(5)")) {
                    consoleOutput.textContent += "Factorial of 5 is: 120\n\n[Process completed successfully in 0.04s]";
                } else if (code.includes("print")) {
                    // Extract strings inside print statements or show generic success
                    consoleOutput.textContent += "Simulated Output:\nHello, Developer! Welcome to Python programming.\nEven squares: [4, 16, 36, 64, 100]\n\n[Process completed successfully in 0.05s]";
                } else {
                    consoleOutput.textContent += "Code executed successfully with no print output.\n\n[Process completed successfully in 0.03s]";
                }
            }, 500);
        });
    }

    // Smooth scroll for nav links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
});
