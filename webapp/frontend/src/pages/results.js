import React, { useState, useEffect } from 'react';
import MediaCard from '../components/imagecard';
import DrawerAppBar from '../components/navbar';
import CircularProgress from '@mui/material/CircularProgress';
import Box from '@mui/material/Box';
import { useLocation } from 'react-router-dom';
import Button from '@mui/material/Button';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';

const Res = () => {
  const [captions, setCaptions] = useState({});
  const [loading, setLoading] = useState(true);
  const location = useLocation();
  const { uploadedImages } = location.state || {};

  useEffect(() => {
    fetch('http://127.0.0.1:8000/captions')
      .then((response) => response.json())
      .then((data) => {
        setCaptions(data.captions);
        setLoading(false);
      })
      .catch((error) => {
        console.error('Error fetching captions:', error);
        setLoading(false);
      });
  }, []);

  const [storedImages, setStoredImages] = useState([]);
  useEffect(() => {
    const retrievedImages = [];
    for (let index = 0; index < localStorage.length; index++) {
      const key = localStorage.key(index);
      if (key && key.startsWith('imageData_')) {
        const imageUrl = localStorage.getItem(key);
        if (imageUrl) {
          retrievedImages.push(imageUrl);
        }
      }
    }
    setStoredImages(retrievedImages);
  }, []);

  // Check if uploadedImages exist and merge them with storedImages
  useEffect(() => {
    if (uploadedImages && uploadedImages.length > 0) {
      setStoredImages((prevImages) => [...prevImages, ...uploadedImages]);
    }
  }, [uploadedImages]);


  const renderImages = () => {
    if (!uploadedImages || uploadedImages.length === 0) {
      return <p>No images found</p>;
    }
  
    const handleSaveClick = () => {
      const pdf = new jsPDF();
  
      const elementList = document.getElementsByClassName('resCard');
      const elementsArray = Array.from(elementList);
  
      const promises = elementsArray.map((element, index) => {
        const backgroundImage = element.querySelector('.MuiCardMedia-root');
        return html2canvas(backgroundImage).then((canvas) => {
          const imgData = canvas.toDataURL('image/png');
          const imgWidth = 210;
          const imgHeight = (canvas.height * imgWidth) / canvas.width;
  
          if (index !== 0) {
            pdf.addPage(); 
          }
  
          pdf.addImage(imgData, 'PNG', 0, 0, imgWidth, imgHeight);
  
          const textContent = element.querySelector('.MuiTypography-body2').innerText;
  
          const textLines = pdf.splitTextToSize(textContent, imgWidth - 20); // Split text based on the width
  
          pdf.text(textLines, 10, imgHeight + 20);
        });
      });
  
      Promise.all(promises).then(() => {
        pdf.save('Report.pdf'); // Save the PDF after all elements are added
      });
    };
  
    return (
      <div>
      <div id="mediaGallery" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        {uploadedImages.map((image, index) => {
          const imageUrl = URL.createObjectURL(image);
          const filename = `temp_image_${index}.png`;
          const caption = captions[filename];
  
          return (
            <div class = 'resCard' style={{ marginRight: '5px' }} key={index}>
              <MediaCard
                src={imageUrl}
                desc={caption}
                title={`Image ${index + 1}`}
              />
            </div>
          );
        })}
      </div>
      <div style={{ textAlign: 'center', marginTop: '20px' }}>
        <Button variant="contained" color="primary" onClick={handleSaveClick}>
          Save Report
        </Button>
      </div>
      </div>
      
    );
  };
  

  return (
    <div>
      <DrawerAppBar />
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        {loading ? (
          <Box sx={{ display: 'block', textAlign: 'center', marginTop: '50px' }}>
            <h2>Waiting for results...</h2>
            <CircularProgress />
          </Box>
        ) : (
          renderImages()
        )}
    </div>

    </div >
  );
};

export default Res;

