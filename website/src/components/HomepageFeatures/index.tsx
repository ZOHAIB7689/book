import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  image: string;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Embodied Intelligence',
    image: require('@site/static/img/physical-ai.jpg').default,
    description: (
      <>
        Learn how intelligence emerges from the interaction between cognitive processes,
        physical body, and environment. Understand morphological computation and
        sensorimotor integration in humanoid systems.
      </>
    ),
  },
  {
    title: 'Modern Robotics Technologies',
    image: require('@site/static/img/robotics-technology.jpg').default,
    description: (
      <>
        Master ROS 2 (Humble/Iron), Gazebo simulation, NVIDIA Isaac Sim,
        and Vision-Language-Action systems for humanoid robotics development.
      </>
    ),
  },
  {
    title: 'Complete Learning Path',
    image: require('@site/static/img/humanoid robotics.png').default,
    description: (
      <>
        From foundational concepts of Physical AI to advanced autonomous
        humanoid implementation. 9 comprehensive chapters with exercises and examples.
      </>
    ),
  },
];

function Feature({title, image, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <img src={image} className={styles.featureSvg} role="img" alt={title} />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
