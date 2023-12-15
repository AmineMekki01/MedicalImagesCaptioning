import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemText from '@mui/material/ListItemText';
import { styled } from '@mui/material/styles';
import MediaCard from '../components/imagecard';
import SwipeableTextMobileStepper from '../components/xrayCarousel'
import CssBaseline from '@mui/material/CssBaseline';
import DrawerAppBar from '../components/navbar';


const mediaCards = [
    <MediaCard
        src={process.env.PUBLIC_URL + '/images/image1.png'}
        desc="A 3-year-old child with visual difficulties. Axial FLAIR image show a supra-sellar lesion extending to the temporal lobes along the optic tracts (arrows) with moderate mass effect, compatible with optic glioma. FLAIR hyperintensity is also noted in the left mesencephalon from additional tumoral involvement"
        title="X-ray chest"
    /> ,
    <MediaCard
    src={process.env.PUBLIC_URL + '/images/image1.png'}
    desc="A 3-year-old child with visual difficulties. Axial FLAIR image show a supra-sellar lesion extending to the temporal lobes along the optic tracts (arrows) with moderate mass effect, compatible with optic glioma. FLAIR hyperintensity is also noted in the left mesencephalon from additional tumoral involvement"
    title="X-ray chest"
    /> ,
    <MediaCard
    src={process.env.PUBLIC_URL + '/images/image1.png'}
    desc="A 3-year-old child with visual difficulties. Axial FLAIR image show a supra-sellar lesion extending to the temporal lobes along the optic tracts (arrows) with moderate mass effect, compatible with optic glioma. FLAIR hyperintensity is also noted in the left mesencephalon from additional tumoral involvement"
    title="X-ray chest"
    /> 
]
const CustomList = styled('ul')({
    listStyle: 'none',
    paddingLeft: 0,
});

const CustomBullet = styled('span')({
    display: 'inline-block',
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    backgroundColor: '#1976d2',
    marginRight: '8px',
});


const About = () => {
    return (
        <div>
            <DrawerAppBar/>
            <Box sx={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'flex-start' }}>
                <div style = {{marginRight : '15px' , marginLeft : '15px'}}>
                    <Typography variant="body1" paragraph sx={{ marginBottom: '20px', lineHeight: '1.6', marginTop: '20px' }}>
                        As Machine Learning Engineers, we understand the challenge of labeling photos.<br />
                        <b>Imagine Radiologists having to  read medical images and write interpretations.</b>
                        Typically, this process can take 5-20 minutes just to review a report. <br /> A doctor might
                        have to assess 40-100 patient images. <br /><br />
                        <b>An automated method to caption medical images could save valuable time for
                            doctors.</b>
                    </Typography>
                    <Typography variant="h6" sx={{ marginBottom: '10px' }}>
                        Who are we?
                    </Typography>
                    <List sx={{ maxWidth: '300px', marginBottom: '20px' }}>
                        <ListItem>
                            <CustomBullet />
                            <ListItemText primary="Amine MEKKI" />
                        </ListItem>
                        <ListItem>
                            <CustomBullet />
                            <ListItemText primary="Mohamed EL YOUSFI ALAOUI" />
                        </ListItem>
                        <ListItem>
                            <CustomBullet />
                            <ListItemText primary="Kaiyuan GONG" />
                        </ListItem>
                        <ListItem>
                            <CustomBullet />
                            <ListItemText primary="Ahmed SIDKI" />
                        </ListItem>
                    </List>
                </div>
                <SwipeableTextMobileStepper mediaCards={mediaCards} />

            </Box>


        </div>
    );
}

export default About;