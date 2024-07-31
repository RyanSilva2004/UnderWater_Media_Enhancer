<body>
    <h1>Underwater Image Enhancement</h1>
    <p>This repository contains a comprehensive solution for enhancing underwater images using various Digital Image Processing (DIP) techniques. The primary goal is to improve the visibility and quality of underwater images by reducing haze, enhancing contrast, and correcting color distortions.</p>
    <p>Below is an example of an original underwater image and its enhanced version using the techniques described above:</p>
    <img src="https://github.com/user-attachments/assets/1c5b0b75-3091-4756-bce4-0d2003f7b660" align="center" />
<br>
<br>
    <h2>Features</h2>
    <ul>
        <li><strong>Dehazing:</strong> Utilizes the Dark Channel Prior (DCP) technique to effectively remove haze from underwater images, enhancing visibility and clarity.</li>
        <li><strong>Color Correction:</strong> Adjusts the color balance to counteract the blue/green cast typical in underwater images, restoring natural colors.</li>
        <li><strong>Contrast Enhancement:</strong> Enhances image contrast to bring out details and improve overall image quality.</li>
        <li><strong>Edge Preservation:</strong> Ensures that edges and fine details are preserved during the enhancement process.</li>
    </ul>
<br>
    <h2>Techniques and Concepts</h2>
<br>
    <h3>1. Dark Channel Prior (DCP)</h3>
    <p>The Dark Channel Prior technique is used for dehazing underwater images. This method is based on the observation that in most non-hazy outdoor images, at least one color channel has some pixels with very low intensities in any local region. For underwater images, the DCP helps in estimating the thickness of the haze and effectively removes it.</p>
<br>
    <h3>2. Color Balance Adjustment</h3>
    <p>Underwater images often suffer from a blue/green cast due to the absorption and scattering of light in the water. To correct this, the program adjusts the color balance by equalizing the histograms of the red, green, and blue channels, thus restoring natural colors.</p>
<br>
    <h3>3. Contrast Limited Adaptive Histogram Equalization (CLAHE)</h3>
    <p>CLAHE is used to enhance the contrast of the images. This technique operates on small regions in the image, called tiles, and enhances the contrast of each tile. It ensures that the histogram of the output region approximately matches the histogram specified by the user, preventing over-amplification of noise.</p>
<br>
    <h3>4. Edge Preservation Smoothing</h3>
    <p>To enhance the image while preserving edges, edge-preserving smoothing filters such as the Bilateral Filter are used. This filter smooths the image while keeping the edges sharp, which is crucial for maintaining the details in underwater images.</p>
<br>
    <h2>Implemented Digital Image Processing Techniques</h2>
    <p>This project utilizes a variety of Digital Image Processing (DIP) techniques to enhance underwater images:</p>
    <ul>
        <li><strong>Image Enhancement:</strong> Core techniques to improve the visual quality of underwater images.</li>
        <li><strong>Histogram Analysis:</strong> Understanding and manipulating the intensity distribution of the image.</li>
        <li><strong>Point Processing Techniques:</strong>
            <ul>
                <li>Histogram Equalization: Redistributing intensity values to enhance contrast.</li>
            </ul>
        </li>
        <li><strong>Spatial Domain Enhancement:</strong> Direct manipulation of pixel values to enhance image quality.</li>
        <li><strong>Filtering:</strong>
            <ul>
                <li>Non-Linear Domain Filtering: Reducing noise while preserving edges.</li>
            </ul>
        </li>
        <li><strong>Color Image Processing:</strong> Adjusting color balance and correcting color casts.</li>
        <li><strong>Dark Channel Prior (DCP) Approach:</strong> Dehazing underwater images to improve clarity.</li>
    </ul>
<br>
    <h2>Usage</h2>
    <ol>
        <li><strong>Clone the repository:</strong></li>
    </ol>
    <pre><code>
    git clone https://github.com/yourusername/underwater-image-enhancement.git
    cd underwater-image-enhancement
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
<br>
    <h2>Contributing</h2>
    <p>Contributions are welcome! Please feel free to submit a Pull Request.</p>
<br>
    <h2>License</h2>
    <p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>
<br>
    <h2>Acknowledgements</h2>
    <ul>
        <li>The Dark Channel Prior technique is based on the paper by Kaiming He, Jian Sun, and Xiaoou Tang.</li>
        <li>The CLAHE method is inspired by the work on Adaptive Histogram Equalization by Karel Zuiderveld.</li>
        <li>Edge-preserving smoothing techniques are derived from the Bilateral Filter by Carlo Tomasi and Roberto Manduchi.</li>
    </ul>
</body>
</html>
