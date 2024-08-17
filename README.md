<body>
    <h1>🌊 Open Source - Underwater Media Enhancer</h1>
    <p>Welcome to the <strong>Underwater Media Enhancement</strong> repository! This project provides a robust solution for enhancing underwater images and videos using a range of Digital Image Processing (DIP) techniques. Our mission is to improve the visibility and quality of underwater imagery by tackling common issues like haze, poor contrast, and color distortion.</p>
    <p>Take a look at this side-by-side comparison of an original underwater image and its enhanced version using our techniques:</p>
    <img src="https://github.com/user-attachments/assets/1c5b0b75-3091-4756-bce4-0d2003f7b660" align="center" />
    <br><br>
    <h2>📸 Python Application UI</h2>
    <p>Here’s a screenshot of the <strong>Underwater Media Enhancer</strong> application in action:</p>
    <img src="https://github.com/user-attachments/assets/b3cdf5fe-b6be-4d18-a5e0-d6810a7d0de1" align="center" />
    <br><br>
    <h2>🖥️ Running the Python Application</h2>
    <p>You can use the Python applications <code>UWME_UI.py</code> and <code>UWME.py</code> or the executable <code>Underwater Media Enhancer v0.1.exe</code> to enhance both images and videos. These tools provide a comprehensive solution for underwater media enhancement.</p>
    <br>
    <h2>✨ Features</h2>
    <ul>
        <li><strong>Dehazing:</strong> Implemented using the Dark Channel Prior (DCP) technique, this feature effectively reduces haze, significantly improving image clarity.</li>
        <li><strong>Color Correction:</strong> Balances colors to counter the blue/green tint typical of underwater photos, restoring natural hues.</li>
        <li><strong>Contrast Enhancement:</strong> Boosts contrast to bring out details and enrich overall image quality.</li>
        <li><strong>Edge Preservation:</strong> Maintains sharpness and fine details during the enhancement process using advanced filters.</li>
    </ul>
    <br>
    <h2>🔍 Techniques and Concepts</h2>
    <h3>1. Dark Channel Prior (DCP)</h3>
    <p>The DCP technique is essential for dehazing underwater images. This method leverages the observation that in most non-hazy outdoor images, at least one color channel has some pixels with very low intensities in any local region. In underwater images, DCP helps estimate haze thickness and remove it effectively.</p>
    <br>
    <h3>2. Color Balance Adjustment</h3>
    <p>Due to light absorption and scattering in water, underwater images often have a blue/green cast. Our color balance adjustment technique corrects this by equalizing the histograms of the red, green, and blue channels, restoring the natural colors.</p>
    <br>
    <h3>3. Contrast Limited Adaptive Histogram Equalization (CLAHE)</h3>
    <p>CLAHE enhances image contrast by working on small regions called tiles. This ensures that the histogram of each tile matches a desired distribution, avoiding over-amplification of noise while improving image detail.</p>
    <br>
    <h3>4. Edge Preservation Smoothing</h3>
    <p>We use edge-preserving smoothing filters like the Bilateral Filter to smooth images without blurring edges, which is crucial for maintaining the fine details in underwater images.</p>
    <br>
    <h2>🔧 Implemented Digital Image Processing Techniques</h2>
    <p>Here's a snapshot of the Digital Image Processing techniques employed in this project:</p>
    <ul>
        <li><strong>Image Enhancement:</strong> Core techniques to improve the visual quality of underwater images and videos.</li>
        <li><strong>Histogram Analysis:</strong> Tools for understanding and manipulating the intensity distribution of images.</li>
        <li><strong>Point Processing Techniques:</strong>
            <ul>
                <li>Histogram Equalization: Enhances contrast by redistributing intensity values.</li>
            </ul>
        </li>
        <li><strong>Spatial Domain Enhancement:</strong> Direct manipulation of pixel values to improve image quality.</li>
        <li><strong>Filtering:</strong>
            <ul>
                <li>Non-Linear Domain Filtering: Reduces noise while preserving edges.</li>
            </ul>
        </li>
        <li><strong>Color Image Processing:</strong> Adjusts color balance and corrects color casts.</li>
        <li><strong>Dark Channel Prior (DCP) Approach:</strong> Dehazes underwater images to improve clarity.</li>
    </ul>
    <br>
    <h2>🚀 Getting Started</h2>
<ol>
    <li><strong>Clone the repository:</strong></li>
</ol>
<pre><code>
git clone https://github.com/RyanSilva2004/UnderWater_Media_Enhancer.git
cd UnderWater_Media_Enhancer
</code></pre>
<ol start="2">
    <li><strong>Install the required dependencies:</strong></li>
</ol>
<pre><code>
pip install -r requirements.txt
</code></pre>
<ol start="3">
    <li><strong>Run the Jupyter Notebook:</strong></li>
</ol>
<p>Open the Jupyter Notebook <code>Under_Water_Image_Enhancement.ipynb</code> and follow the step-by-step instructions to enhance your underwater images.</p>
<ol start="4">
    <li><strong>Run the Python Applications:</strong></li>
</ol>
<p>For interactive usage for images & videos , you can run the Python applications:</p>
<pre><code>
python UWME_UI.py
</code></pre>
<ol start="5">
    <li><strong>Use the Executable File:</strong></li>
</ol>
<p>If you prefer a quick access option, run the executable <code>Underwater Media Enhancer v0.1.exe</code>. This will provide a user-friendly interface for both image and video enhancement.</p>
    <br>
    <h2>🖥️ Running the Python Application</h2>
    <p>You can also run the Python applications <code>UWME_UI.py</code> or <code>UWME.py</code> for interactive usage, or use the executable <code>Underwater Media Enhancer v0.1.exe</code> for quick access. These tools support both image and video enhancement.</p>
    <br>
    <h2>🤝 Contributing</h2>
    <p>We welcome all contributions to make this project even better! Whether it's adding new features, improving existing algorithms, or optimizing the code, your input is highly valued. To contribute, fork the repository, create a feature branch, and submit a Pull Request. Let's collaborate to push the boundaries of what's possible in underwater image and video enhancement!</p>
    <br>
    <h2>📜 License</h2>
    <p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>
    <br>
    <h2>🙏 Acknowledgements</h2>
    <ul>
        <li>The Dark Channel Prior technique is based on the paper by Kaiming He, Jian Sun, and Xiaoou Tang.</li>
        <li>The CLAHE method is inspired by Karel Zuiderveld's work on Adaptive Histogram Equalization.</li>
        <li>Edge-preserving smoothing techniques are derived from the Bilateral Filter by Carlo Tomasi and Roberto Manduchi.</li>
    </ul>
</body>
