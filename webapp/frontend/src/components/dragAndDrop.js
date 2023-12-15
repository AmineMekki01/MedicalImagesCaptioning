import React, { useState } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import { Button } from '@mui/material';
import Slider from '@mui/material/Slider';
import { useNavigate } from "react-router-dom";


const temperatures = [
  {
    value: 0.5,
    label: '0.5',
  },
  {
    value: 1,
    label: '1.0',
  },
  {
    value: 1.5,
    label: '1.5',
  },
];

function valuetext(value) {
  return `${value}`;
}


const DragAndDropImages = () => {
  const navigate = useNavigate();
  const [imageFiles, setImageFiles] = useState([]);
  const [temperature, setTemperature] = useState(1.0);

  const handleDrop = (e) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    const selectedFiles = Array.from(files).slice(0, 5);
    const imageFilesArray = selectedFiles.filter(
      file => file.type === 'image/png' || file.type === 'image/jpeg'
    );
    setImageFiles(prevFiles => [...prevFiles, ...imageFilesArray]);

  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleFileInputChange = (e) => {
    const files = e.target.files;
    const selectedFiles = Array.from(files).slice(0, 3);
    const imageFilesArray = selectedFiles.filter(
      file => file.type === 'image/png' || file.type === 'image/jpeg'
    );
    setImageFiles(prevFiles => [...prevFiles, ...imageFilesArray]);
    
  };



  const downloadFiles = async () => {

    try {
      const formData = new FormData();
      imageFiles.forEach((file) => {
        formData.append('images', file);
      });

      formData.append('temperature', temperature);
      const response = await fetch('http://127.0.0.1:8000/download', {
        method: 'POST',
        body: formData,
      });

      console.log(imageFiles.length);
      navigate('/results', { state: { uploadedImages: imageFiles } });

    } catch (error) {
      console.error('Error downloading files:', error);
    }
  };

  const handleTemperatureChange = (event, newValue) => {
    setTemperature(newValue); 
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <Typography variant="h6" gutterBottom>
        Temperature:
      </Typography>
      <Box sx={{ width: '300px', textAlign: 'center' }}>
        <Slider
          aria-label="Always visible"
          defaultValue={1.0}
          getAriaValueText={valuetext}
          onChange={handleTemperatureChange}
          step={0.1}
          min={0.5}
          max={1.5} 
          temperatures={temperatures}
          valueLabelDisplay="on"
        />
      </Box>
      <Box
        sx={{
          maxWidth: '500px',
          margin: 'auto',
          paddingTop: '20px',
          border: '2px dashed #aaa',
          borderRadius: '5px',
          textAlign: 'center',
        }}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >

        <input
          accept=".png,.jpg,.jpeg"
          type="file"
          style={{ display: 'none' }}
          onChange={handleFileInputChange}
          multiple
        />
        <Typography variant="h5" sx={{ margin: '20px 0' }}>
          Drag and Drop to Upload up to 3 Images (PNG/JPG)
        </Typography>
        {imageFiles.length > 0 && (
          <div>
            <h2>Uploaded Images:</h2>
            <div style={{ display: 'flex', flexWrap: 'wrap' }}>
              {imageFiles.map((file, index) => {
                const imageUrl = URL.createObjectURL(file);
                localStorage.setItem(`imageData_${index}`, imageUrl);
                return (
                  <div key={index} style={{ margin: '10px' }}>
                    <img src={imageUrl} alt={`Uploaded ${index}`} style={{ maxWidth: '150px' }} />
                  </div>
                );
              })}
            </div>
            <Button variant="contained" onClick={downloadFiles}>Generate report</Button>
          </div>
        )}

      </Box>
    </Box>
  );
};

export default DragAndDropImages;



