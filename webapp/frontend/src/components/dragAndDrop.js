import React, { useState } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';

const DragAndDropImages = () => {
  const [imageFiles, setImageFiles] = useState([]);

  const handleDrop = (e) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    const selectedFiles = Array.from(files).slice(0, 5); // Limit to 5 files
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
    const selectedFiles = Array.from(files).slice(0, 5); // Limit to 5 files
    const imageFilesArray = selectedFiles.filter(
      file => file.type === 'image/png' || file.type === 'image/jpeg'
    );
    setImageFiles(prevFiles => [...prevFiles, ...imageFilesArray]);
  };

  return (
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
        // You can also add the "multiple" attribute to allow selecting multiple files using file dialog
        // But it doesn't limit the number of files like the drag and drop functionality
      />
      <Typography variant="h5" sx={{ margin: '20px 0' }}>
        Drag and Drop or Click to Upload up to 5 Images (PNG/JPG)
      </Typography>
      {imageFiles.length > 0 && (
        <div>
          <h2>Uploaded Images:</h2>
          <div style={{ display: 'flex', flexWrap: 'wrap' }}>
            {imageFiles.map((file, index) => (
              <div key={index} style={{ margin: '10px' }}>
                <img src={URL.createObjectURL(file)} alt={`Uploaded ${index}`} style={{ maxWidth: '150px' }} />
              </div>
            ))}
          </div>
        </div>
      )}
    </Box>
  );
};

export default DragAndDropImages;



