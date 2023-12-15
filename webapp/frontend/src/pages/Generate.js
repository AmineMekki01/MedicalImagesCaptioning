import Typography from '@mui/material/Typography';
import DragAndDropImages from '../components/dragAndDrop';
import DrawerAppBar from '../components/navbar';
import { useEffect } from 'react';

const Generate = () => {
    useEffect(() => {
        localStorage.clear(); // Clears the local storage when component mounts
    }, []);
    return (
        <Typography>
            <DrawerAppBar/>
            <DragAndDropImages/>
        </Typography>
    );
}

export default Generate;