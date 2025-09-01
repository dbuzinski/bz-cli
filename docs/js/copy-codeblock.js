// Copy code blocks to clipboard
document.addEventListener('DOMContentLoaded', function() {
  // Find all pre elements
  const codeBlocks = document.querySelectorAll('pre');
  
  codeBlocks.forEach(function(pre) {
    // Create copy button with clipboard icon
    const copyButton = document.createElement('button');
    copyButton.className = 'copy-button';
    copyButton.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
      </svg>
    `;
    copyButton.style.cssText = `
      position: absolute;
      top: 8px;
      right: 8px;
      background-color: #374151;
      color: #f9fafb;
      border: none;
      border-radius: 4px;
      padding: 6px;
      cursor: pointer;
      opacity: 0.7;
      transition: all 0.15s ease-in-out;
      z-index: 10;
      display: flex;
      align-items: center;
      justify-content: center;
      width: 28px;
      height: 28px;
    `;
    
    // Add hover effect
    copyButton.addEventListener('mouseenter', function() {
      this.style.opacity = '1';
    });
    
    copyButton.addEventListener('mouseleave', function() {
      this.style.opacity = '0.7';
    });
    
    // Add click handler
    copyButton.addEventListener('click', function() {
      // Get the code content from the code element inside pre
      const codeElement = pre.querySelector('code');
      let code = codeElement ? codeElement.textContent : pre.textContent;
      
      // Remove extra whitespace and normalize line endings
      code = code.trim();
      
      // Copy to clipboard
      navigator.clipboard.writeText(code).then(function() {
        // Show success feedback
        const originalHTML = copyButton.innerHTML;
        copyButton.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="20,6 9,17 4,12"></polyline>
          </svg>
        `;
        copyButton.style.backgroundColor = '#10b981';
        
        setTimeout(function() {
          copyButton.innerHTML = originalHTML;
          copyButton.style.backgroundColor = '#374151';
        }, 2000);
      }).catch(function(err) {
        console.error('Failed to copy: ', err);
        const originalHTML = copyButton.innerHTML;
        copyButton.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        `;
        copyButton.style.backgroundColor = '#ef4444';
        
        setTimeout(function() {
          copyButton.innerHTML = originalHTML;
          copyButton.style.backgroundColor = '#374151';
        }, 2000);
      });
    });
    
    // Add button to pre element
    pre.style.position = 'relative';
    pre.appendChild(copyButton);
  });
});
